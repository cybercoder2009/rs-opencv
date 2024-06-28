use std::error::Error;
use std::path::Path;
use opencv::core::{bitwise_and, split, Point, Scalar, Size, TermCriteria, Vector, BORDER_REFLECT};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{
    adaptive_threshold, cvt_color, dilate, get_structuring_element, pyr_mean_shift_filtering,
    COLOR_BGR2Lab, COLOR_Lab2BGR, ADAPTIVE_THRESH_MEAN_C, MORPH_RECT, THRESH_BINARY,
};
use opencv::prelude::Mat;
use opencv::ximgproc::anisotropic_diffusion;

pub fn convert(file_path: &str) -> Result<(), Box<dyn Error>> {

    let path = Path::new(file_path);
    let folder = path.parent().unwrap().to_str().unwrap();
    let filename = path.file_name().unwrap().to_str().unwrap();

    /* load img */
    let mat_bgr = imread(path.to_str().unwrap(), IMREAD_COLOR)?;
    
    let mat_lab = bgr_to_lab(&mat_bgr)?;

    /* base */
    let mut mat_0 = segment_colors(&mat_lab)?;
    // opencv::highgui::imshow("segmented", &mat_0)?;
    mat_0 = lab_to_bgr(&mat_0)?;
    
    /* border */
    let mut mat_1 = anisotropic_blur(&mat_lab)?;
    // opencv::highgui::imshow("blurred", &mat_1)?;
    mat_1 = gray_from_lab(&mat_1)?;
    // opencv::highgui::imshow("grayscaled", &mat_1)?;
    mat_1 = grayscaled_to_edged(&mat_1)?;
    // opencv::highgui::imshow("edged", &mat_1)?;
    
    /* merge */
    let output = combine_base_and_edge(&mat_0, &mat_1)?;
    // opencv::highgui::imshow("output", &output)?;
    let path_write = format!("{}/{}", folder, filename.replace(".", ".nft."));
    imwrite(&path_write, &output, &Vector::default())?;

    // opencv::highgui::wait_key(0)?;
    Ok(())
}

/*
 * BGR image -> Lab img
 */
fn bgr_to_lab(input: &Mat) -> Result<Mat, Box<dyn Error>> {
    let mut output = Mat::default();
    cvt_color(&input, &mut output, COLOR_BGR2Lab, 0)?;
    Ok(output)
}

/*
 * Lab img -> BGR image
 */
fn lab_to_bgr(input: &Mat) -> Result<Mat, Box<dyn Error>> {
    let mut output = Mat::default();
    cvt_color(&input, &mut output, COLOR_Lab2BGR, 0)?;
    Ok(output)
}

/// Extracts the lightness channel from the Lab image.
fn gray_from_lab(input: &Mat) -> Result<Mat, Box<dyn Error>> {
    let mut channels = Vector::<Mat>::new();
    split(input, &mut channels)?;
    // Extract the L channel (index 0) from the Lab image
    let output = channels.get(0)?;
    Ok(output)
}

/*
 * grayscaled image -> edged image
 */ 
fn grayscaled_to_edged(input: &Mat) -> Result<Mat, Box<dyn Error>> {
    let max_binary_value = 255.0;
    let mut edges = Mat::default();
    adaptive_threshold(
        input,
        &mut edges,
        max_binary_value,
        ADAPTIVE_THRESH_MEAN_C,
        THRESH_BINARY,
        9,
        9.0,
    )?;

    // Dilate the edges, i.e. make them less prominent.
    let mut output = Mat::default();
    let kernel = get_structuring_element(MORPH_RECT, Size::new(3, 3), Point::new(-1, -1))?;
    let anchor = Point::new(-1, -1);
    let iterations = 1;
    dilate(
        &edges,
        &mut output,
        &kernel,
        anchor,
        iterations,
        BORDER_REFLECT,
        Scalar::default(),
    )?;
    Ok(output)
}

/**
 * OpenCV provides an image processing function based on the Mean Shift algorithm. 
 * This function is primarily used for image smoothing and segmentation, especially 
 * excelling at removing noise while preserving edges and details in the image. 
 * It is a non-parametric clustering method that performs smoothing and segmentation 
 * by finding high-density regions of data points in both the spatial and color spaces.
 */
fn segment_colors(input: &Mat) -> Result<Mat, Box<dyn Error>> {
    let spatial_radius = 10.0;
    let color_radius = 20.0;
    let max_pyramid_level = 1;
    let term_criteria = TermCriteria::default()?;
    let mut output = Mat::default();
    pyr_mean_shift_filtering(
        &input,
        &mut output,
        spatial_radius,
        color_radius,
        max_pyramid_level,
        term_criteria,
    )?;
    Ok(output)
}

/**
 * Anisotropic Blur is an image processing technique used to apply non-uniform blurring to an image. 
 * This method adjusts the direction and degree of blur based on the local characteristics of the image, 
 * such as edge direction and intensity, in order to preserve edge details.
 */
fn anisotropic_blur(input: &Mat) -> Result<Mat, Box<dyn Error>> {
    let mut output = Mat::default();
    let conductance = 0.1;
    let time_step = 0.05;
    let num_iterations = 10;
    anisotropic_diffusion(
        &input,
        &mut output,
        time_step,
        conductance,
        num_iterations,
    )?;
    Ok(output)
}

fn combine_base_and_edge(
    base: &Mat,
    edge: &Mat,
) -> Result<Mat, Box<dyn Error>> {
    let mut output = Mat::default();
    bitwise_and(base, base, &mut output, edge)?;
    Ok(output)
}