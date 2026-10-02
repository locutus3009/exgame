// SPDX-License-Identifier: MIT

use crate::wgpu;

/// Window + surface configuration for [`crate::run`].
#[derive(Clone, Debug)]
pub struct Config {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub present_mode: wgpu::PresentMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            title: "melies".to_string(),
            width: 800,
            height: 800,
            present_mode: wgpu::PresentMode::AutoVsync,
        }
    }
}

impl Config {
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

/// Builder for [`Config`] (project convention: 4+ field structs get a builder).
#[derive(Clone, Debug, Default)]
pub struct ConfigBuilder {
    title: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    vsync: Option<bool>,
}

impl ConfigBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// `true` → `AutoVsync`, `false` → `AutoNoVsync`.
    pub fn vsync(mut self, vsync: bool) -> Self {
        self.vsync = Some(vsync);
        self
    }

    pub fn build(self) -> Config {
        let d = Config::default();
        let present_mode = match self.vsync {
            Some(false) => wgpu::PresentMode::AutoNoVsync,
            _ => wgpu::PresentMode::AutoVsync,
        };
        Config {
            title: self.title.unwrap_or(d.title),
            width: self.width.unwrap_or(d.width),
            height: self.height.unwrap_or(d.height),
            present_mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let c = Config::default();
        assert_eq!(c.title, "melies");
        assert_eq!((c.width, c.height), (800, 800));
        assert_eq!(c.present_mode, wgpu::PresentMode::AutoVsync);
    }

    #[test]
    fn builder_overrides_and_vsync_mapping() {
        let c = Config::builder()
            .title("newton")
            .size(640, 480)
            .vsync(false)
            .build();
        assert_eq!(c.title, "newton");
        assert_eq!((c.width, c.height), (640, 480));
        assert_eq!(c.present_mode, wgpu::PresentMode::AutoNoVsync);
    }
}
