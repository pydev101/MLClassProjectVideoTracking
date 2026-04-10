//! Head click labeler: open an image, click heads, export HDF5 mask + jet heatmap PNG.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use eframe::egui;
use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::Array2;
use rfd::FileDialog;
use serde::Serialize;

/// Gaussian blur σ in pixels (full-resolution mask). Increase for wider heat blobs.
const GAUSSIAN_SIGMA_PX: f32 = 12.0;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_maximized(true)
            .with_title("Head click labeler"),
        ..Default::default()
    };
    eframe::run_native(
        "labeling",
        options,
        Box::new(|cc| Ok(Box::new(LabelApp::new(cc)))),
    )
}

struct LabelApp {
    texture: Option<egui::TextureHandle>,
    image_path: Option<PathBuf>,
    image_wh: Option<[u32; 2]>,
    clicks: Vec<(u32, u32)>,
    status: String,
    out_root: PathBuf,
    /// Multiplier on top of “fit entire viewport” scale (scroll wheel).
    zoom: f32,
    /// Screen-space offset of image center from viewport center (middle-mouse pan).
    pan: egui::Vec2,
    /// Sorted images from “Open folder…”, for Prev/Next.
    batch_folder: Option<PathBuf>,
    batch_paths: Vec<PathBuf>,
    batch_index: usize,
    /// Last successful “Create dataset” export (shown in the right panel).
    last_output: Option<LastOutput>,
    /// True after loading or editing clicks until a successful export.
    needs_save: bool,
    /// Preview of the last exported heatmap PNG (right panel).
    heatmap_preview: Option<egui::TextureHandle>,
}

/// Paths written by the most recent export.
struct LastOutput {
    base: String,
    out_root: PathBuf,
    manifest: PathBuf,
    h5: PathBuf,
    heat: PathBuf,
    source: PathBuf,
}

impl LabelApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let out = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("labeled_data");
        Self {
            texture: None,
            image_path: None,
            image_wh: None,
            clicks: Vec::new(),
            status: "Open an image, click heads, then Create dataset.".into(),
            out_root: out,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            batch_folder: None,
            batch_paths: Vec::new(),
            batch_index: 0,
            last_output: None,
            needs_save: false,
            heatmap_preview: None,
        }
    }

    fn load_image_path(&mut self, ctx: &egui::Context, path: PathBuf) {
        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let w = rgba.width();
                let h = rgba.height();
                let size = [w as usize, h as usize];
                let color = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                let tex = ctx.load_texture(path.display().to_string(), color, Default::default());
                self.texture = Some(tex);
                self.image_path = Some(path);
                self.image_wh = Some([w, h]);
                self.clicks.clear();
                self.zoom = 1.0;
                self.pan = egui::Vec2::ZERO;
                self.heatmap_preview = None;
                self.needs_save = false;
                self.update_status_after_load(w, h);
            }
            Err(e) => {
                self.status = format!("Failed to open image: {e}");
            }
        }
    }

    fn update_status_after_load(&mut self, w: u32, h: u32) {
        if self.batch_paths.is_empty() {
            self.status = format!("Loaded {w}×{h}. Clicks: 0");
        } else {
            self.status = format!(
                "{} / {} — {}×{}. Clicks: 0",
                self.batch_index + 1,
                self.batch_paths.len(),
                w,
                h
            );
        }
    }

    fn format_clicks_status(&self, detail: Option<&str>) -> String {
        let n = self.clicks.len();
        let batch = if self.batch_paths.is_empty() {
            None
        } else {
            Some((self.batch_index + 1, self.batch_paths.len()))
        };
        match (batch, detail) {
            (None, None) => format!("Clicks: {n}"),
            (None, Some(d)) => format!("Clicks: {n} ({d})"),
            (Some((i, t)), None) => format!("{i}/{t} — Clicks: {n}"),
            (Some((i, t)), Some(d)) => format!("{i}/{t} — Clicks: {n} ({d})"),
        }
    }

    fn load_heatmap_preview(&mut self, ctx: &egui::Context, path: &Path) {
        match image::open(path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let w = rgba.width();
                let h = rgba.height();
                let size = [w as usize, h as usize];
                let color = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                let id = format!("heatmap_preview_{}", path.display());
                let tex = ctx.load_texture(id, color, Default::default());
                self.heatmap_preview = Some(tex);
            }
            Err(_) => {
                self.heatmap_preview = None;
            }
        }
    }

    fn save_dataset(&mut self, ctx: &egui::Context) -> Result<(), String> {
        let path = self.image_path.as_ref().ok_or("No image loaded")?;
        let [w, h] = self.image_wh.ok_or("No image size")?;

        let out_root = self.out_root.clone();
        let gt_dir = out_root.join("ground_truth");
        let gt_h5_dir = gt_dir.join("h5");
        let gt_json_dir = gt_dir.join("json");
        let src_dir = out_root.join("source");
        let heat_dir = out_root.join("heatmap");
        fs::create_dir_all(&gt_h5_dir).map_err(|e| e.to_string())?;
        fs::create_dir_all(&gt_json_dir).map_err(|e| e.to_string())?;
        fs::create_dir_all(&src_dir).map_err(|e| e.to_string())?;
        fs::create_dir_all(&heat_dir).map_err(|e| e.to_string())?;

        let stem = sanitize_stem(
            path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("image"),
        );
        let ts = Utc::now().format("%Y%m%dT%H%M%SZ");
        let base = format!("{stem}_{ts}");

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{e}"))
            .unwrap_or_else(|| ".png".into());

        let mut mask = Array2::<u8>::zeros((h as usize, w as usize));
        for &(x, y) in &self.clicks {
            let xi = x.min(w - 1) as usize;
            let yi = y.min(h - 1) as usize;
            mask[[yi, xi]] = 1;
        }

        let h5_path = gt_h5_dir.join(format!("{base}.h5"));
        write_h5_mask(&h5_path, &mask, GAUSSIAN_SIGMA_PX)?;

        let heat_path = heat_dir.join(format!("{base}.png"));
        write_jet_heatmap_png(&heat_path, &mask, w, h, GAUSSIAN_SIGMA_PX)?;

        let dest_src = src_dir.join(format!("{base}{ext}"));
        fs::copy(path, &dest_src).map_err(|e| e.to_string())?;

        let manifest = Manifest {
            schema_version: 1,
            source_image: path.canonicalize().unwrap_or_else(|_| path.clone()),
            dimensions_hw: [h, w],
            total_heads: self.clicks.len(),
            clicks_xy_image_space: self
                .clicks
                .iter()
                .map(|&(x, y)| Click { x, y })
                .collect(),
            files: Files {
                manifest_json: PathBuf::from("ground_truth")
                    .join("json")
                    .join(format!("{base}.json")),
                ground_truth_h5: PathBuf::from("ground_truth")
                    .join("h5")
                    .join(format!("{base}.h5")),
                source_image_copy: PathBuf::from("source").join(format!("{base}{ext}")),
                heatmap_png: PathBuf::from("heatmap").join(format!("{base}.png")),
            },
            gaussian_sigma_pixels: GAUSSIAN_SIGMA_PX,
        };
        let json = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
        let manifest_path = gt_json_dir.join(format!("{base}.json"));
        fs::write(&manifest_path, json).map_err(|e| e.to_string())?;

        self.last_output = Some(LastOutput {
            base: base.clone(),
            out_root: out_root.clone(),
            manifest: manifest_path,
            h5: h5_path,
            heat: heat_path.clone(),
            source: dest_src,
        });
        self.needs_save = false;
        self.load_heatmap_preview(ctx, &heat_path);
        self.status = format!("Saved {base} under {}", out_root.display());
        Ok(())
    }
}

impl eframe::App for LabelApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar")
            .frame(
                egui::Frame::side_top_panel(&ctx.style())
                    .inner_margin(egui::Margin::symmetric(8.0, 6.0)),
            )
            .show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open image…").clicked() {
                    if let Some(file) = FileDialog::new().pick_file() {
                        self.batch_folder = None;
                        self.batch_paths.clear();
                        self.batch_index = 0;
                        self.load_image_path(ctx, file);
                    }
                }
                if ui.button("Open folder…").clicked() {
                    if let Some(dir) = FileDialog::new().pick_folder() {
                        match scan_folder_images(&dir) {
                            Ok(paths) => {
                                if paths.is_empty() {
                                    self.status = "No supported images in that folder.".into();
                                } else {
                                    self.batch_folder = Some(dir);
                                    self.batch_paths = paths;
                                    self.batch_index = 0;
                                    let first = self.batch_paths[0].clone();
                                    self.load_image_path(ctx, first);
                                }
                            }
                            Err(e) => self.status = format!("Folder: {e}"),
                        }
                    }
                }
                ui.add_enabled_ui(!self.batch_paths.is_empty(), |ui| {
                    if ui.button("◀ Prev").clicked() && self.batch_index > 0 {
                        self.batch_index -= 1;
                        let p = self.batch_paths[self.batch_index].clone();
                        self.load_image_path(ctx, p);
                    }
                });
                let can_next = !self.batch_paths.is_empty()
                    && self.batch_index + 1 < self.batch_paths.len();
                ui.add_enabled_ui(can_next, |ui| {
                    if ui.button("Next ▶").clicked() {
                        self.batch_index += 1;
                        let p = self.batch_paths[self.batch_index].clone();
                        self.load_image_path(ctx, p);
                    }
                });
                if !self.batch_paths.is_empty() {
                    ui.label(format!(
                        "{} / {}",
                        self.batch_index + 1,
                        self.batch_paths.len()
                    ));
                }
                if ui.button("Undo last click").clicked() {
                    self.clicks.pop();
                    self.needs_save = !self.clicks.is_empty();
                    if self.image_wh.is_some() {
                        self.status = self.format_clicks_status(None);
                    }
                }
                if ui.button("Clear clicks").clicked() {
                    self.clicks.clear();
                    self.needs_save = false;
                    if self.image_wh.is_some() {
                        self.status = self.format_clicks_status(None);
                    }
                }
                if ui.button("Create dataset…").clicked() {
                    match self.save_dataset(ctx) {
                        Ok(()) => {}
                        Err(e) => self.status = format!("Error: {e}"),
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Output root:");
                let mut s = self.out_root.display().to_string();
                if ui.text_edit_singleline(&mut s).changed() {
                    self.out_root = PathBuf::from(s);
                }
                if ui.button("Browse…").clicked() {
                    if let Some(d) = FileDialog::new().pick_folder() {
                        self.out_root = d;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.label(egui::RichText::new("Scroll: zoom  ·  Middle drag: pan  ·  Left click: label").weak());
                if self.needs_save && !self.clicks.is_empty() {
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("Needs save — export (Create dataset)")
                            .color(egui::Color32::RED),
                    );
                }
            });
        });

        egui::SidePanel::right("last_export")
            .default_width(300.0)
            .min_width(220.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Heatmap preview");
                ui.separator();
                if let Some(ht) = &self.heatmap_preview {
                    let max_w = ui.available_width();
                    let sz = ht.size_vec2();
                    let scale = (max_w / sz.x).min(1.0);
                    ui.add(
                        egui::Image::new((ht.id(), sz * scale))
                            .sense(egui::Sense::hover()),
                    );
                } else {
                    ui.label(
                        egui::RichText::new("Export once to show the generated heatmap here.")
                            .weak(),
                    );
                }
                ui.add_space(8.0);
                ui.heading("Last export");
                ui.separator();
                match &self.last_output {
                    None => {
                        ui.label(
                            egui::RichText::new("Create a dataset to see file paths here.")
                                .weak(),
                        );
                    }
                    Some(lo) => {
                        ui.label(egui::RichText::new("Base id").strong());
                        ui.monospace(&lo.base);
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new("Output root").strong());
                        ui.monospace(lo.out_root.display().to_string());
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new("Files").strong());
                        let rows = [
                            ("Manifest", &lo.manifest),
                            ("Ground truth (.h5)", &lo.h5),
                            ("Source copy", &lo.source),
                            ("Heatmap (.png)", &lo.heat),
                        ];
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for (label, path) in rows {
                                ui.label(egui::RichText::new(label).small());
                                ui.monospace(path.display().to_string());
                                ui.add_space(4.0);
                            }
                        });
                    }
                }
            });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::central_panel(&ctx.style())
                    .inner_margin(egui::Margin::ZERO)
                    .fill(egui::Color32::from_rgb(30, 30, 30)),
            )
            .show(ctx, |ui| {
                let available = ui.max_rect();

                if let (Some(tex), Some([w, h])) = (&self.texture, self.image_wh) {
                    let tex_sz = tex.size_vec2();
                    let fit = (available.width() / tex_sz.x).min(available.height() / tex_sz.y);
                    let fit = if fit.is_finite() && fit > 0.0 {
                        fit
                    } else {
                        1.0
                    };

                    if let Some(hover) = ctx.pointer_hover_pos() {
                        if available.contains(hover) {
                            let dy = ctx.input(|i| i.raw_scroll_delta.y);
                            if dy != 0.0 {
                                let factor = 1.0 + dy * 0.0015;
                                self.zoom = (self.zoom * factor).clamp(0.05, 80.0);
                            }
                        }
                    }

                    if ctx.input(|i| i.pointer.button_down(egui::PointerButton::Middle)) {
                        self.pan += ctx.input(|i| i.pointer.delta());
                    }

                    let display = tex_sz * fit * self.zoom;
                    let center = available.center() + self.pan;
                    let rect = egui::Rect::from_center_size(center, display);

                    let img_id = ui.id().with("image_canvas");
                    let img_response = ui.interact(rect, img_id, egui::Sense::click());

                    let painter = ui.painter_at(available);
                    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                    painter.image(tex.id(), rect, uv, egui::Color32::WHITE);

                    for &(ix, iy) in &self.clicks {
                        let cx = rect.min.x + (ix as f32 + 0.5) / w as f32 * rect.width();
                        let cy = rect.min.y + (iy as f32 + 0.5) / h as f32 * rect.height();
                        let r = (4.0 * fit * self.zoom).max(2.0).min(24.0);
                        painter.circle_stroke(
                            egui::pos2(cx, cy),
                            r,
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 255, 136)),
                        );
                    }

                    if img_response.clicked_by(egui::PointerButton::Primary)
                        && !ctx.input(|i| i.pointer.button_down(egui::PointerButton::Middle))
                    {
                        if let Some(pos) = img_response.interact_pointer_pos() {
                            let u = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
                            let v = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0);
                            let px = (u * w as f32).round() as u32;
                            let py = (v * h as f32).round() as u32;
                            let px = px.min(w.saturating_sub(1));
                            let py = py.min(h.saturating_sub(1));
                            self.clicks.push((px, py));
                            self.needs_save = true;
                            self.status = self.format_clicks_status(Some(&format!(
                                "last {px},{py}"
                            )));
                        }
                    }
                } else {
                    ui.label("No image loaded.");
                }
            });
    }
}

/// 1D Gaussian kernel, odd length, normalized.
fn scan_folder_images(dir: &Path) -> Result<Vec<PathBuf>, String> {
    const EXT: &[&str] = &["jpg", "jpeg", "png", "bmp", "gif", "webp", "tif", "tiff"];
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|x| x.to_str())
                    .map(|s| EXT.iter().any(|&e| e.eq_ignore_ascii_case(s)))
                    .unwrap_or(false)
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let r = (sigma * 3.0_f32).ceil().max(1.0) as i32;
    let denom = 2.0 * sigma * sigma;
    let mut k: Vec<f32> = (-r..=r)
        .map(|i| (-((i as f32).powi(2)) / denom).exp())
        .collect();
    let sum: f32 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

#[inline]
fn sample_zero(data: &[f32], width: usize, height: usize, x: isize, y: isize) -> f32 {
    if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
        return 0.0;
    }
    data[y as usize * width + x as usize]
}

/// Separable Gaussian blur with zero-padded edge sampling.
fn gaussian_blur_separable(data: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let kernel = gaussian_kernel_1d(sigma);
    let r = (kernel.len() / 2) as isize;
    let mut tmp = vec![0.0_f32; width * height];
    let mut out = vec![0.0_f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                let ox = x as isize + ki as isize - r;
                acc += kw * sample_zero(data, width, height, ox, y as isize);
            }
            tmp[y * width + x] = acc;
        }
    }

    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                let oy = y as isize + ki as isize - r;
                acc += kw * sample_zero(&tmp, width, height, x as isize, oy);
            }
            out[y * width + x] = acc;
        }
    }
    out
}

/// Classic “jet” colormap (matplotlib-style piecewise RGB), `t` in [0, 1].
fn jet_colormap_rgb(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = (1.5 - (4.0 * (t - 0.75).abs()).min(1.5)).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * (t - 0.5).abs()).min(1.5)).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * (t - 0.25).abs()).min(1.5)).clamp(0.0, 1.0);
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    ]
}

fn sanitize_stem(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    s.chars().take(120).collect()
}

fn write_h5_mask(path: &Path, mask: &Array2<u8>, sigma: f32) -> Result<(), String> {
    use hdf5_pure::{AttrValue, FileBuilder};

    let shape = mask.shape();
    let h = shape[0] as u64;
    let w = shape[1] as u64;
    let data = mask
        .as_slice()
        .ok_or("internal: mask must be contiguous row-major array")?;

    let mut builder = FileBuilder::new();
    builder
        .create_dataset("mask")
        .with_shape(&[h, w])
        .with_u8_data(data)
        .set_attr(
            "description",
            AttrValue::String(
                "Binary head click mask: 1 at clicked pixel, 0 elsewhere".into(),
            ),
        )
        .set_attr(
            "axes",
            AttrValue::String("row_major: dim0=height (y), dim1=width (x)".into()),
        )
        .set_attr("sigma_heatmap_px", AttrValue::F64(f64::from(sigma)));
    builder.write(path).map_err(|e| e.to_string())
}

fn write_jet_heatmap_png(
    path: &Path,
    mask: &Array2<u8>,
    width: u32,
    height: u32,
    sigma: f32,
) -> Result<(), String> {
    let w = width as usize;
    let h = height as usize;
    let mut flat: Vec<f32> = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            flat[y * w + x] = mask[[y, x]] as f32;
        }
    }
    let blurred = gaussian_blur_separable(&flat, w, h, sigma);
    let mut mx: f32 = 0.0;
    for &v in &blurred {
        mx = mx.max(v);
    }
    let mut rgb: RgbImage = ImageBuffer::new(width, height);
    for y in 0..h {
        for x in 0..w {
            let t = if mx > 0.0 {
                blurred[y * w + x] / mx
            } else {
                0.0
            };
            let c = jet_colormap_rgb(t);
            rgb.put_pixel(x as u32, y as u32, Rgb(c));
        }
    }
    rgb.save(path).map_err(|e| e.to_string())?;
    Ok(())
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    source_image: PathBuf,
    dimensions_hw: [u32; 2],
    total_heads: usize,
    clicks_xy_image_space: Vec<Click>,
    files: Files,
    gaussian_sigma_pixels: f32,
}

#[derive(Serialize)]
struct Click {
    x: u32,
    y: u32,
}

#[derive(Serialize)]
struct Files {
    manifest_json: PathBuf,
    ground_truth_h5: PathBuf,
    source_image_copy: PathBuf,
    heatmap_png: PathBuf,
}
