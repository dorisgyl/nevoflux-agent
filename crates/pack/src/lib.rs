//! NevoFlux Pack protocol engine — pure logic (no daemon/tokio deps).
//! Platform side effects go through [`host::PackHost`]. Module declarations
//! are added by their respective tasks (B2–B8).

pub mod capability;
pub mod error;
pub mod host;
pub mod lifecycle;
pub mod manifest;
pub mod paths;
pub mod receipt;

pub use host::PackHost;
pub use manifest::Manifest;
pub use receipt::Receipt;

pub use error::{PackError, PackResult};
pub use paths::ResolvedPaths;
