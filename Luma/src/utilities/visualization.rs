use std::io::Write;
use std::fs::File;
use std::path::Path;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Data point for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
    pub label: Option<String>,
}

/// Series of data for plotting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSeries {
    pub name: String,
    pub points: Vec<DataPoint>,
    pub color: Option<String>,
    pub line_style: Option<String>,
}

/// Represents a complete plot with multiple data series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plot {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub series: Vec<DataSeries>,
    pub width: usize,
    pub height: usize,
    pub show_legend: bool,
    pub grid: bool,
}

impl Default for Plot {
    fn default() -> Self {
        Plot {
            title: "Untitled Plot".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            series: Vec::new(),
            width: 800,
            height: 600,
            show_legend: true,
            grid: true,
        }
    }
}

impl Plot {
    pub fn new(title: &str) -> Self {
        Plot {
            title: title.to_string(),
            ..Default::default()
        }
    }
    
    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }
    
    /// Generate a simple SVG representation of the plot
    pub fn to_svg(&self) -> String {
        let mut svg = String::new();
        
        // SVG header
        svg.push_str(&format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.width, self.height, self.width, self.height
        ));
        
        // Background
        svg.push_str(&format!(
            r#"<rect width="{}" height="{}" fill="white"/>"#,
            self.width, self.height
        ));
        
        // Title
        svg.push_str(&format!(
            r#"<text x="{}" y="30" font-family="sans-serif" font-size="24px" text-anchor="middle">{}</text>"#,
            self.width / 2, self.title
        ));
        
        // Plot area
        let margin = 60;
        let plot_width = self.width - 2 * margin;
        let plot_height = self.height - 2 * margin;
        
        // Find data ranges
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        
        for series in &self.series {
            for point in &series.points {
                min_x = min_x.min(point.x);
                max_x = max_x.max(point.x);
                min_y = min_y.min(point.y);
                max_y = max_y.max(point.y);
            }
        }
        
        // Add some padding to the ranges
        let x_padding = (max_x - min_x) * 0.05;
        let y_padding = (max_y - min_y) * 0.05;
        min_x -= x_padding;
        max_x += x_padding;
        min_y -= y_padding;
        max_y += y_padding;
        
        // If we have a single point or all points have same value, create a reasonable range
        if min_x == max_x {
            min_x -= 1.0;
            max_x += 1.0;
        }
        if min_y == max_y {
            min_y -= 1.0;
            max_y += 1.0;
        }
        
        // Draw grid if enabled
        if self.grid {
            // Horizontal grid lines
            for i in 0..=5 {
                let plot_height_f64 = plot_height as f64;
                let y_pos = margin + (plot_height_f64 - plot_height_f64 * i as f64 / 5.0) as usize;
                svg.push_str(&format!(
                    "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#DDDDDD\" stroke-width=\"1\"/>",
                    margin, y_pos, margin + plot_width, y_pos
                ));
                
                let y_value = min_y + (max_y - min_y) * i as f64 / 5.0;
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" font-family="sans-serif" font-size="12px" text-anchor="end">{:.2}</text>"#,
                    margin - 5, y_pos + 4, y_value
                ));
            }
            
            // Vertical grid lines
            for i in 0..=5 {
                let plot_width_f64 = plot_width as f64;
                let x_pos = margin + (plot_width_f64 * i as f64 / 5.0) as usize;
                svg.push_str(&format!(
                    "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#DDDDDD\" stroke-width=\"1\"/>",
                    x_pos, margin, x_pos, margin + plot_height
                ));
                
                let x_value = min_x + (max_x - min_x) * i as f64 / 5.0;
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" font-family="sans-serif" font-size="12px" text-anchor="middle">{:.2}</text>"#,
                    x_pos, margin + plot_height + 20, x_value
                ));
            }
        }
        
        // Draw axes
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2"/>"#,
            margin, margin, margin, margin + plot_height
        ));
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2"/>"#,
            margin, margin + plot_height, margin + plot_width, margin + plot_height
        ));
        
        // Draw axis labels
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="sans-serif" font-size="16px" text-anchor="middle">{}</text>"#,
            margin + plot_width / 2, margin + plot_height + 40, self.x_label
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="sans-serif" font-size="16px" text-anchor="middle" transform="rotate(-90, {}, {})">{}</text>"#,
            margin - 40, margin + plot_height / 2, margin - 40, margin + plot_height / 2, self.y_label
        ));
        
        // Draw each data series
        let colors = vec!["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01", "#46BDC6", "#7B29ED"];
        
        for (idx, series) in self.series.iter().enumerate() {
            let color = series.color.clone().unwrap_or_else(|| colors[idx % colors.len()].to_string());
            
            // Plot line if we have more than one point
            if series.points.len() > 1 {
                let mut path = String::new();
                
                for (i, point) in series.points.iter().enumerate() {
                    let x = margin + ((point.x - min_x) / (max_x - min_x) * plot_width as f64) as usize;
                    let y = margin + plot_height - ((point.y - min_y) / (max_y - min_y) * plot_height as f64) as usize;
                    
                    if i == 0 {
                        path.push_str(&format!("M {} {}", x, y));
                    } else {
                        path.push_str(&format!(" L {} {}", x, y));
                    }
                }
                
                svg.push_str(&format!(
                    r#"<path d="{}" fill="none" stroke="{}" stroke-width="2"/>"#,
                    path, color
                ));
            }
            
            // Plot points
            for point in &series.points {
                let x = margin + ((point.x - min_x) / (max_x - min_x) * plot_width as f64) as usize;
                let y = margin + plot_height - ((point.y - min_y) / (max_y - min_y) * plot_height as f64) as usize;
                
                svg.push_str(&format!(
                    r#"<circle cx="{}" cy="{}" r="4" fill="{}" stroke="black" stroke-width="1"/>"#,
                    x, y, color
                ));
                
                if let Some(label) = &point.label {
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" font-family="sans-serif" font-size="10px" text-anchor="middle">{}</text>"#,
                        x, y - 8, label
                    ));
                }
            }
        }
        
        // Draw legend if enabled and we have multiple series
        if self.show_legend && self.series.len() > 1 {
            let legend_x = margin + plot_width - 150;
            let legend_y = margin + 30;
            
            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="140" height="{}" fill="white" stroke="black" stroke-width="1" fill-opacity="0.8"/>"#,
                legend_x, legend_y, 20 * self.series.len() + 10
            ));
            
            for (idx, series) in self.series.iter().enumerate() {
                let color = series.color.clone().unwrap_or_else(|| colors[idx % colors.len()].to_string());
                let y_pos = legend_y + 20 + idx * 20;
                
                svg.push_str(&format!(
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
                    legend_x + 10, y_pos, legend_x + 30, y_pos, color
                ));
                svg.push_str(&format!(
                    r#"<circle cx="{}" cy="{}" r="3" fill="{}" stroke="black" stroke-width="1"/>"#,
                    legend_x + 20, y_pos, color
                ));
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" font-family="sans-serif" font-size="12px">{}</text>"#,
                    legend_x + 40, y_pos + 4, series.name
                ));
            }
        }
        
        // Close SVG
        svg.push_str("</svg>");
        svg
    }
    
    /// Save the plot as an SVG file
    pub fn save_svg(&self, path: &str) -> Result<(), String> {
        let svg_content = self.to_svg();
        let mut file = File::create(path).map_err(|e| format!("Failed to create SVG file: {}", e))?;
        file.write_all(svg_content.as_bytes()).map_err(|e| format!("Failed to write SVG data: {}", e))?;
        Ok(())
    }
    
    /// Save the plot as JSON for external rendering
    pub fn save_json(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize plot to JSON: {}", e))?;
        
        let mut file = File::create(path).map_err(|e| format!("Failed to create JSON file: {}", e))?;
        file.write_all(json.as_bytes()).map_err(|e| format!("Failed to write JSON data: {}", e))?;
        
        Ok(())
    }
}

/// Create a data series from raw vectors of x and y values
pub fn series_from_vectors(name: &str, x_values: &[f64], y_values: &[f64]) -> DataSeries {
    assert_eq!(x_values.len(), y_values.len(), "x and y vectors must have the same length");
    
    let mut points = Vec::with_capacity(x_values.len());
    for i in 0..x_values.len() {
        points.push(DataPoint {
            x: x_values[i],
            y: y_values[i],
            label: None,
        });
    }
    
    DataSeries {
        name: name.to_string(),
        points,
        color: None,
        line_style: None,
    }
}

/// Convenience function to plot a simple line from vectors
pub fn plot_line(title: &str, x_values: &[f64], y_values: &[f64], output_path: &str) -> Result<(), String> {
    let series = series_from_vectors("Data", x_values, y_values);
    
    let mut plot = Plot::new(title);
    plot.x_label = "X".to_string();
    plot.y_label = "Y".to_string();
    plot.add_series(series);
    
    // Determine file extension and save accordingly
    if output_path.to_lowercase().ends_with(".json") {
        plot.save_json(output_path)
    } else {
        plot.save_svg(output_path)
    }
}

/// Plot a histogram from a vector of values
pub fn plot_histogram(title: &str, values: &[f64], bins: usize, output_path: &str) -> Result<(), String> {
    if values.is_empty() {
        return Err("Cannot create histogram from empty data".to_string());
    }
    
    // Find min and max
    let min_val = values.iter().fold(f64::MAX, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::MIN, |a, &b| a.max(b));
    
    // Create bins
    let bin_width = (max_val - min_val) / bins as f64;
    let mut bin_counts = vec![0; bins];
    
    // Count values in each bin
    for &value in values {
        let bin_idx = ((value - min_val) / bin_width).min((bins - 1) as f64).max(0.0) as usize;
        bin_counts[bin_idx] += 1;
    }
    
    // Create data points (x at bin center)
    let mut x_values = Vec::with_capacity(bins);
    let mut y_values = Vec::with_capacity(bins);
    
    for i in 0..bins {
        x_values.push(min_val + bin_width * (i as f64 + 0.5));
        y_values.push(bin_counts[i] as f64);
    }
    
    // Create plot
    plot_line(title, &x_values, &y_values, output_path)
}

/// Plot a scatter plot from two vectors
pub fn plot_scatter(title: &str, x_values: &[f64], y_values: &[f64], output_path: &str) -> Result<(), String> {
    assert_eq!(x_values.len(), y_values.len(), "x and y vectors must have the same length");
    
    let series = series_from_vectors("Data", x_values, y_values);
    
    let mut plot = Plot::new(title);
    plot.x_label = "X".to_string();
    plot.y_label = "Y".to_string();
    plot.add_series(series);
    
    // Determine file extension and save accordingly
    if output_path.to_lowercase().ends_with(".json") {
        plot.save_json(output_path)
    } else {
        plot.save_svg(output_path)
    }
}

/// Plot training metrics (loss, accuracy) over epochs
pub fn plot_training_metrics(
    metrics: &HashMap<String, Vec<f64>>, 
    epochs: usize, 
    output_path: &str
) -> Result<(), String> {
    let mut plot = Plot::new("Training Metrics");
    plot.x_label = "Epoch".to_string();
    plot.y_label = "Value".to_string();
    
    // Create x-values for epochs (0-indexed)
    let x_values: Vec<f64> = (0..epochs).map(|i| i as f64).collect();
    
    // Add each metric as a separate series
    for (metric_name, values) in metrics {
        if values.len() != epochs {
            return Err(format!(
                "Expected {} values for metric '{}', but got {}", 
                epochs, metric_name, values.len()
            ));
        }
        
        let series = series_from_vectors(metric_name, &x_values, values);
        plot.add_series(series);
    }
    
    // Determine file extension and save accordingly
    if output_path.to_lowercase().ends_with(".json") {
        plot.save_json(output_path)
    } else {
        plot.save_svg(output_path)
    }
}

/// Legacy function for backward compatibility
pub fn visualize_data(data: &[f64], file: &mut std::fs::File) {
    let x_values: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();
    
    let svg = match plot_line("Data Visualization", &x_values, data, "temp_vis.svg") {
        Ok(_) => {
            match std::fs::read_to_string("temp_vis.svg") {
                Ok(content) => content,
                Err(_) => format!("Failed to read SVG file. Data: {:?}", data),
            }
        },
        Err(e) => format!("Failed to generate plot: {}. Data: {:?}", e, data),
    };
    
    writeln!(file, "{}", svg).unwrap_or_else(|_| {
        eprintln!("Failed to write SVG to file");
    });
    
    // Clean up temp file if it exists
    if Path::new("temp_vis.svg").exists() {
        let _ = std::fs::remove_file("temp_vis.svg");
    }
}

/// Legacy function for backward compatibility
pub fn plot_data(data: &[f64], output_path: &str) -> Result<(), String> {
    let x_values: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();
    plot_line("Data Plot", &x_values, data, output_path)
}

/// C API for data visualization
#[no_mangle]
pub extern "C" fn luma_plot_data(data: *const f64, len: i32, output_path: *const std::os::raw::c_char) -> i32 {
    if data.is_null() || output_path.is_null() || len <= 0 {
        return -1;
    }
    
    let path_str = unsafe { 
        match std::ffi::CStr::from_ptr(output_path).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };
    
    let data_slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    
    match plot_data(data_slice, path_str) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_plot_creation() {
        let mut plot = Plot::new("Test Plot");
        plot.x_label = "X Axis".to_string();
        plot.y_label = "Y Axis".to_string();
        
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_values = vec![2.0, 4.0, 1.0, 5.0, 3.0];
        
        let series = series_from_vectors("Test Series", &x_values, &y_values);
        plot.add_series(series);
        
        let svg = plot.to_svg();
        assert!(svg.contains("Test Plot"));
        assert!(svg.contains("X Axis"));
        assert!(svg.contains("Y Axis"));
    }
    
    #[test]
    fn test_plot_save() {
        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_values = vec![2.0, 4.0, 1.0, 5.0, 3.0];
        
        let temp_file = "test_plot.svg";
        plot_line("Test Save", &x_values, &y_values, temp_file).unwrap();
        
        assert!(Path::new(temp_file).exists());
        
        // Clean up
        fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_histogram() {
        let values = vec![1.0, 1.5, 2.0, 2.2, 2.7, 3.0, 3.1, 3.5, 4.0, 4.5, 4.7, 5.0];
        
        let temp_file = "test_histogram.svg";
        plot_histogram("Test Histogram", &values, 5, temp_file).unwrap();
        
        assert!(Path::new(temp_file).exists());
        
        // Clean up
        fs::remove_file(temp_file).unwrap();
    }
}