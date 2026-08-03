//! NevoFlux Native Agent CLI
//!
//! Entry points:
//! - `nevoflux-agent` - Proxy mode (Native Messaging bridge)
//! - `nevoflux-agent --daemon` - Core daemon
//! - `nevoflux-agent --mcp` - MCP server mode
//! - `nevoflux-agent --status` - Show daemon status
//! - `nevoflux-agent --stop` - Stop daemon
//! - `nevoflux-agent config <action>` - Configuration management
//! - `nevoflux-agent setup` - Interactive setup wizard

mod cli;
mod completions;
mod logging;

use clap::Parser;
use cli::PackAction;
use cli::{AccountAction, Cli, Commands, ConfigAction};
use fs2::FileExt;
use nevoflux_storage::Storage;
use std::fs::File;
use std::path::PathBuf;

/// Get the data directory for NevoFlux.
///
/// Platform-specific locations (via `directories` crate):
/// - Linux: ~/.local/share/nevoflux/
/// - macOS: ~/Library/Application Support/com.nevoflux.nevoflux/
/// - Windows: %APPDATA%\nevoflux\nevoflux\data\
fn get_data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("NEVOFLUX_DATA_DIR") {
        return PathBuf::from(dir);
    }

    directories::ProjectDirs::from("com", "nevoflux", "nevoflux")
        .map(|dirs| dirs.data_dir().to_path_buf())
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".nevoflux")
        })
}

/// Ensure the data directory exists.
#[allow(dead_code)]
fn ensure_data_dir() -> std::io::Result<PathBuf> {
    let dir = get_data_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Daemon status result.
#[derive(Debug)]
enum DaemonStatus {
    /// Daemon is running.
    Running { port: u16, pid: u32 },
    /// Daemon files exist but process is not running.
    Stale { port: u16, pid: u32 },
    /// Daemon is not running.
    NotRunning,
}

/// Check if the daemon is running.
fn check_daemon_status() -> DaemonStatus {
    let data_dir = get_data_dir();
    let port_file = data_dir.join("daemon.port");
    let pid_file = data_dir.join("daemon.pid");

    // Read port file
    let port = match std::fs::read_to_string(&port_file) {
        Ok(s) => match s.trim().parse::<u16>() {
            Ok(p) => p,
            Err(_) => return DaemonStatus::NotRunning,
        },
        Err(_) => return DaemonStatus::NotRunning,
    };

    // Read PID file
    let pid = match std::fs::read_to_string(&pid_file) {
        Ok(s) => match s.trim().parse::<u32>() {
            Ok(p) => p,
            Err(_) => return DaemonStatus::Stale { port, pid: 0 },
        },
        Err(_) => return DaemonStatus::Stale { port, pid: 0 },
    };

    // Check if process is running
    if is_process_running(pid) {
        DaemonStatus::Running { port, pid }
    } else {
        DaemonStatus::Stale { port, pid }
    }
}

/// Check if a process is running.
#[cfg(unix)]
fn is_process_running(pid: u32) -> bool {
    use std::process::Command;
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(windows)]
fn is_process_running(pid: u32) -> bool {
    use std::process::Command;
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {}", pid)])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

/// Run the status command.
fn run_status() {
    match check_daemon_status() {
        DaemonStatus::Running { port, pid } => {
            println!("Daemon is running");
            println!("  port: {}", port);
            println!("  pid: {}", pid);
        }
        DaemonStatus::Stale { port, pid } => {
            println!("Daemon is not running (stale files found)");
            println!("  port file: {}", port);
            println!("  pid file: {}", pid);
            println!("Run 'nevoflux-agent --stop' to clean up.");
        }
        DaemonStatus::NotRunning => {
            println!("Daemon is not running");
        }
    }
}

/// Stop the running daemon.
fn stop_daemon() -> std::io::Result<()> {
    let data_dir = get_data_dir();
    let port_file = data_dir.join("daemon.port");
    let pid_file = data_dir.join("daemon.pid");
    let lock_file = data_dir.join("daemon.lock");

    // Try to read PID and send SIGTERM
    if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            if is_process_running(pid) {
                #[cfg(unix)]
                {
                    use std::process::Command;
                    let _ = Command::new("kill")
                        .args(["-TERM", &pid.to_string()])
                        .output();

                    // Wait briefly for graceful shutdown
                    std::thread::sleep(std::time::Duration::from_millis(500));

                    // Force kill if still running
                    if is_process_running(pid) {
                        let _ = Command::new("kill")
                            .args(["-KILL", &pid.to_string()])
                            .output();
                    }
                }
                #[cfg(windows)]
                {
                    use std::process::Command;
                    let _ = Command::new("taskkill")
                        .args(["/F", "/PID", &pid.to_string()])
                        .output();
                }
            }
        }
    }

    // Clean up files
    let _ = std::fs::remove_file(&port_file);
    let _ = std::fs::remove_file(&pid_file);
    let _ = std::fs::remove_file(&lock_file);

    Ok(())
}

/// Run the stop command.
fn run_stop() {
    match check_daemon_status() {
        DaemonStatus::Running { port: _, pid } => {
            println!("Stopping daemon (PID {})...", pid);
            if let Err(e) = stop_daemon() {
                eprintln!("Error stopping daemon: {}", e);
                std::process::exit(1);
            }
            println!("Daemon stopped.");
        }
        DaemonStatus::Stale { .. } => {
            println!("Cleaning up stale daemon files...");
            if let Err(e) = stop_daemon() {
                eprintln!("Error cleaning up: {}", e);
                std::process::exit(1);
            }
            println!("Cleanup complete.");
        }
        DaemonStatus::NotRunning => {
            println!("Daemon is not running.");
        }
    }
}

/// Acquire the daemon lock file.
///
/// - `managed=false`: uses `daemon.lock` (dev / manually-started daemon)
/// - `managed=true`: uses `daemon-managed.lock` (proxy-spawned daemon)
fn acquire_daemon_lock(managed: bool) -> std::io::Result<File> {
    let data_dir = ensure_data_dir()?;
    let lock_name = if managed {
        "daemon-managed.lock"
    } else {
        "daemon.lock"
    };
    let lock_path = data_dir.join(lock_name);

    let file = File::create(&lock_path)?;
    file.try_lock_exclusive().map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            if managed {
                "Managed daemon is already running"
            } else {
                "Daemon is already running"
            },
        )
    })?;

    Ok(file)
}

/// Run in proxy mode (Native Messaging bridge).
///
/// This bridges between the browser extension (via Native Messaging on stdin/stdout)
/// and the daemon (via TCP). Uses full-duplex communication to allow receiving
/// messages (like cancel requests or browser tool responses) while streaming.
async fn run_proxy(verbose: bool, dev_mode: bool) -> Result<(), Box<dyn std::error::Error>> {
    use nevoflux_bridge::{run_async_proxy, AsyncProxyConfig, BridgeConfig, ConnectionMode};
    use std::io::IsTerminal;
    use std::sync::{Arc, Mutex};
    use tokio::io::{stdin, stdout};

    // Layer 2: refuse to run proxy mode from an interactive terminal. The
    // proxy speaks the Native Messaging binary framing on stdin/stdout, so a
    // human at a tty cannot meaningfully drive it. Refusing here prevents
    // users from accidentally invoking proxy mode when they meant to run a
    // CLI subcommand.
    if std::io::stdin().is_terminal() {
        eprintln!(
            "nevoflux-agent: proxy mode is for browser Native Messaging, not interactive use."
        );
        eprintln!();
        eprintln!("For CLI usage, try:");
        eprintln!("  nevoflux-agent --daemon     start the daemon");
        eprintln!("  nevoflux-agent --mcp        run MCP server");
        eprintln!("  nevoflux-agent --status     show daemon status");
        eprintln!("  nevoflux-agent --stop       stop the daemon");
        eprintln!("  nevoflux-agent --help       full help");
        std::process::exit(2);
    }

    // Layer 1: on Windows, only hide the console if we exclusively own it.
    // GetConsoleProcessList returns the number of processes attached to the
    // current console. When a browser launches us via Native Messaging the
    // OS allocates a fresh console and we are the sole owner (count == 1),
    // so hiding it removes a brief flash. When a user launches us from
    // PowerShell / cmd / Windows Terminal we inherit the parent's console
    // (count >= 2) — hiding it would hide the user's own terminal window.
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Console::{GetConsoleProcessList, GetConsoleWindow};
        use windows_sys::Win32::UI::WindowsAndMessaging::{ShowWindow, SW_HIDE};
        let mut pids = [0u32; 2];
        let count = unsafe { GetConsoleProcessList(pids.as_mut_ptr(), 2) };
        if count == 1 {
            let hwnd = unsafe { GetConsoleWindow() };
            if !hwnd.is_null() {
                unsafe { ShowWindow(hwnd, SW_HIDE) };
            }
        }
    }

    // Ensure data directory exists before writing log files or port/pid files.
    let data_dir = ensure_data_dir().unwrap_or_else(|_| get_data_dir());

    // Initialize logging to file only (stdout/stderr must be silent for Native Messaging)
    let log_file = data_dir.join("proxy.log");
    logging::init_file_only_logging(log_file, verbose);

    let mode = if dev_mode {
        ConnectionMode::Dev
    } else {
        ConnectionMode::Prod
    };

    tracing::debug!("Starting async proxy mode (full-duplex, {:?})", mode);

    // Layer 3: share the spawned-daemon PID with the Ctrl+C branch so we can
    // still kill the daemon if the proxy future is cancelled before it
    // returns normally.
    let pid_slot: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));

    // Optimistic onboarding hint. Read "is a provider configured?" from the
    // same config file the daemon reads (AgentConfig::load does no env
    // merging), so this matches the daemon's authoritative `status.first_run`.
    // When nothing is configured, hand the proxy a `setup_status` message to
    // emit immediately — the sidebar can render "Start Setup" during the
    // daemon's cold boot instead of waiting for it. A configured user gets
    // `None`, leaving the startup sequence unchanged.
    let early_setup_status = match nevoflux_daemon::AgentConfig::load() {
        Ok(cfg) if cfg.llm.has_any_configured_provider() => None,
        _ => Some(serde_json::json!({
            "type": "setup_status",
            "payload": {
                "first_run": true,
                "has_configured_provider": false,
                "optimistic": true,
                "version": env!("CARGO_PKG_VERSION"),
            }
        })),
    };

    let bridge_config = BridgeConfig::new().with_mode(mode).with_data_dir(data_dir);
    let config = AsyncProxyConfig::new()
        .with_bridge(bridge_config)
        .with_spawned_pid_slot(pid_slot.clone())
        .with_early_setup_status(early_setup_status);

    let proxy_fut = run_async_proxy(stdin(), stdout(), config);

    let spawned_pid = tokio::select! {
        result = proxy_fut => result?.spawned_daemon_pid,
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Ctrl+C received in proxy mode, shutting down");
            pid_slot.lock().ok().and_then(|g| *g)
        }
    };

    // In prod mode, clean up the spawned daemon.
    // Zero-file mode: only kill the process. No port/pid/lock files to remove.
    if let Some(pid) = spawned_pid {
        tracing::info!("Proxy shutting down, killing spawned daemon PID {}", pid);
        kill_process(pid);
        // No files to clean up — daemon was launched with --port (zero-file mode)
    }

    Ok(())
}

/// Kill a process by PID. Sends SIGTERM, waits, then SIGKILL if needed.
#[cfg(unix)]
fn kill_process(pid: u32) {
    use std::process::Command;

    // Send SIGTERM
    let _ = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .output();

    // Wait up to 3s for graceful shutdown
    for _ in 0..30 {
        if !is_process_running(pid) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Force kill if still running
    if is_process_running(pid) {
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .output();
    }
}

/// Kill a process by PID on Windows.
#[cfg(windows)]
fn kill_process(pid: u32) {
    use std::process::Command;
    let _ = Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .output();
}

/// Run the MCP server mode.
///
/// This starts an MCP server that communicates via stdio, allowing
/// Claude Code or other MCP clients to use browser automation tools.
async fn run_mcp(verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    use nevoflux_mcp::{create_tools, run_stdio_server, McpServer, McpServerConfig};

    // Initialize logging to stderr (stdout is for MCP protocol)
    logging::init_stderr_logging(verbose, Some("nevoflux=info"));

    logging::log_startup(env!("CARGO_PKG_VERSION"));
    tracing::info!("Starting MCP server mode");

    // Create server with default configuration
    let config = McpServerConfig::default();
    let mut server = McpServer::with_config(config);

    // Register all tools from the tools module
    for tool in create_tools() {
        server.register_tool(tool);
    }

    // Run the stdio server
    run_stdio_server(server).await?;

    Ok(())
}

/// Bind the current process into a Windows Job Object configured with
/// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.
///
/// Why: when the daemon exits (Ctrl+C, force-kill, panic, or the explicit
/// `std::process::exit(0)` at the end of `run_daemon`), Windows tears down
/// its handle table. The Job's last handle is closed, the Job is destroyed,
/// and KILL_ON_JOB_CLOSE causes every assigned subprocess to be terminated.
/// Any child the daemon `spawn`s after this call inherits the Job
/// automatically, so MCP servers, sub-agents, etc. cannot outlive us and
/// keep the listening socket pinned.
///
/// Best-effort: failures are logged and ignored so a permissions edge-case
/// (e.g. an outer Job that disallows nesting on very old Windows) doesn't
/// prevent the daemon from starting.
#[cfg(windows)]
fn assign_self_to_kill_on_close_job() {
    use std::mem;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            tracing::warn!(
                "CreateJobObjectW failed (err={}); orphan children may survive daemon exit",
                std::io::Error::last_os_error()
            );
            return;
        }

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        if SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        ) == 0
        {
            tracing::warn!(
                "SetInformationJobObject failed (err={}); orphan children may survive",
                std::io::Error::last_os_error()
            );
            return;
        }

        if AssignProcessToJobObject(job, GetCurrentProcess()) == 0 {
            tracing::warn!(
                "AssignProcessToJobObject failed (err={}); orphan children may survive",
                std::io::Error::last_os_error()
            );
            return;
        }

        // Deliberately leak the Job handle. We want it open for the entire
        // process lifetime — on exit the kernel closes it, dropping the
        // Job's last reference and firing KILL_ON_JOB_CLOSE.
        let _ = job;
    }
}

#[cfg(not(windows))]
fn assign_self_to_kill_on_close_job() {
    // No-op on Unix-like platforms: tokio + signal-based shutdown handle
    // child termination on those targets. A future enhancement could use
    // `prctl(PR_SET_PDEATHSIG)` per child on Linux.
}

/// Terminate the current process immediately, skipping ALL in-process
/// cleanup.
///
/// `std::process::exit` first runs Rust's runtime cleanup (which flushes
/// stdout) and then, on Windows, `ExitProcess` — which runs DLL detach
/// under the loader lock. With `--verbose` logging across many threads plus
/// the uncancellable `spawn_blocking` LLM-call threads still running at
/// shutdown, either step can deadlock: a thread parked mid-write holds the
/// stdout lock the flush needs, or a force-terminated thread holds a
/// CRT/loader lock that DLL detach then waits on. The symptom is the daemon
/// hanging at the very end of shutdown with Ctrl+C appearing dead.
///
/// `TerminateProcess(GetCurrentProcess())` kills the process without
/// flushing stdout, without DLL detach, and without taking the loader lock,
/// so it cannot deadlock. The leaked kill-on-close Job handle is closed by
/// the kernel as the process dies, so child processes are still reaped.
#[cfg(windows)]
fn force_exit(code: u32) -> ! {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
    // SAFETY: GetCurrentProcess returns a pseudo-handle to the current
    // process; TerminateProcess on it does not return.
    unsafe {
        TerminateProcess(GetCurrentProcess(), code);
    }
    // TerminateProcess on self never returns, but its signature can't prove
    // that. Spin so this function is `!` and we never fall through.
    loop {
        std::hint::spin_loop();
    }
}

/// Unix variant: the Windows `ExitProcess`/loader-lock hang does not apply,
/// and `std::process::exit` is the idiomatic immediate exit.
#[cfg(not(windows))]
fn force_exit(code: u32) -> ! {
    std::process::exit(code as i32);
}

/// Spawn a detached OS thread that force-terminates the process after
/// `grace`, as a last-resort backstop for shutdown.
///
/// Deliberately a plain `std::thread`, NOT a tokio task: if the async
/// shutdown path wedges (an uncancellable `spawn_blocking` call, a Windows
/// `ExitProcess` deadlock, a stdout flush blocked on a paused console),
/// tokio can't schedule a task to rescue us — but an independent OS thread
/// sleeping on a timer is immune to all of it. When the orderly path wins
/// the race it calls [`force_exit`] first and this thread dies with the
/// process, never firing.
fn spawn_force_exit_watchdog(grace: std::time::Duration) {
    std::thread::spawn(move || {
        std::thread::sleep(grace);
        // Reached only if orderly shutdown never terminated us.
        eprintln!("shutdown watchdog: process did not exit within {grace:?}; forcing termination");
        force_exit(0);
    });
}

/// Run the bundled post-update cleanup script once per version.
///
/// On the first daemon start after a version change, spawn the platform cleanup
/// script from `<exe_dir>/cleanup/` to clear stale add-on caches so an update
/// never keeps serving an old cached sidebar. Passes `--no-kill` (the browser is
/// running — never kill the live session) and `--keep-skills` (the skills
/// refresh is handled separately, with consent). Fully best-effort: any failure
/// is logged and ignored, and the version marker advances so it runs at most
/// once per version. Locked cache files (held by the running browser) are simply
/// skipped and take effect on the next launch; Firefox also self-invalidates its
/// startup caches on a build-id change.
fn run_post_update_cleanup(data_dir: &std::path::Path) {
    let current = env!("CARGO_PKG_VERSION");
    let marker = data_dir.join(".last-cleanup-version");
    if std::fs::read_to_string(&marker)
        .map(|s| s.trim() == current)
        .unwrap_or(false)
    {
        return; // already ran for this version
    }

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("post-update cleanup: cannot resolve exe path: {}", e);
            return;
        }
    };
    let Some(cleanup_dir) = exe.parent().map(|d| d.join("cleanup")) else {
        return;
    };

    #[cfg(windows)]
    let script = cleanup_dir.join("nevoflux-cleanup.ps1");
    #[cfg(target_os = "macos")]
    let script = cleanup_dir.join("nevoflux-cleanup.command");
    #[cfg(all(unix, not(target_os = "macos")))]
    let script = cleanup_dir.join("nevoflux-cleanup.sh");

    if script.exists() {
        #[cfg(windows)]
        let mut cmd = {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            let mut c = std::process::Command::new("powershell.exe");
            c.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]);
            c.arg(&script);
            c.args(["-NoKill", "-KeepSkills"]);
            c.creation_flags(CREATE_NO_WINDOW);
            c
        };
        #[cfg(unix)]
        let mut cmd = {
            let mut c = std::process::Command::new("bash");
            c.arg(&script);
            c.args(["--no-kill", "--keep-skills"]);
            c
        };
        // Detach from the native-messaging stdio channel; the cleanup output
        // must never reach the protocol wire.
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());

        match cmd.spawn() {
            Ok(_child) => tracing::info!(
                "post-update cleanup: launched {} (version -> {})",
                script.display(),
                current
            ),
            Err(e) => tracing::warn!("post-update cleanup: failed to spawn: {}", e),
        }
    } else {
        tracing::debug!(
            "post-update cleanup: no bundled script at {}",
            script.display()
        );
    }

    // Advance the marker regardless of outcome: this is a best-effort, once-per
    // version belt-and-suspenders step (Firefox self-invalidates its caches on a
    // build-id change), so we do not want to re-spawn on every daemon start.
    if let Err(e) = std::fs::write(&marker, current) {
        tracing::warn!("post-update cleanup: failed to write version marker: {}", e);
    }
}

/// Run the daemon.
#[allow(clippy::too_many_arguments)]
async fn run_daemon(
    verbose: bool,
    trace: bool,
    port_start: Option<u16>,
    port_end: Option<u16>,
    port: Option<u16>,
    managed: bool,
    headless: bool,
    remote_control: bool,
    http_addr: Option<std::net::SocketAddr>,
    openai_addr: Option<std::net::SocketAddr>,
    mcp_addr: Option<std::net::SocketAddr>,
    acp_addr: Option<std::net::SocketAddr>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Ensure data directory exists first — managed daemons need it for the
    // log file, and SessionManager needs it for the database. This must
    // happen before init_logging() so the log file can actually be created.
    let data_dir = ensure_data_dir()?;

    // Initialize logging — managed daemons write to a log file since the
    // bridge launches them with stderr redirected to null.
    let log_file = if managed {
        Some(data_dir.join("daemon.log"))
    } else {
        None
    };
    logging::init_logging(verbose, log_file);

    // Install the kill-on-close Job binding before spawning any subprocess
    // (MCP servers, sub-agents). Any child created after this point is
    // automatically associated with the Job and dies when we do.
    assign_self_to_kill_on_close_job();

    let trace_enabled = trace
        || std::env::var("NEVOFLUX_TRACE")
            .map(|v| v == "1")
            .unwrap_or(false);
    if trace_enabled {
        tracing::info!("Trace enabled: writing to {}/traces/", data_dir.display());
    }

    logging::log_startup(env!("CARGO_PKG_VERSION"));

    // Ensure config file exists on first launch. This must happen early so
    // users have a config.toml to edit even if the daemon fails later.
    // On Windows in managed mode, stdout/stderr are null, so without this
    // explicit step, a missing config would be silently ignored.
    if let Err(e) = nevoflux_daemon::AgentConfig::load() {
        tracing::warn!("Failed to initialize config file: {}", e);
    }

    // Install bundled default skills on first launch
    match nevoflux_skills::install_default_skills() {
        Ok(0) => {} // already installed or no bundled skills
        Ok(n) => tracing::info!("Installed {} default skill files", n),
        Err(e) => tracing::warn!("Failed to install default skills: {}", e),
    }

    // On a version bump, run the bundled post-update cleanup once so an update
    // never keeps serving a stale cached sidebar. Best-effort; never blocks or
    // fails startup, and never kills the running browser or touches skills.
    run_post_update_cleanup(&data_dir);

    // In managed+port mode the proxy is the lifecycle manager — skip file lock.
    let _lock = if managed && port.is_some() {
        None
    } else {
        match acquire_daemon_lock(managed) {
            Ok(lock) => Some(lock),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                eprintln!("Error: Daemon is already running");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error acquiring lock: {}", e);
                std::process::exit(1);
            }
        }
    };

    // Create session manager with database
    let db_path = get_db_path();
    let session_manager = std::sync::Arc::new(
        nevoflux_daemon::SessionManager::new(db_path.to_str().unwrap_or("nevoflux.db"))
            .expect("Failed to create session manager"),
    );

    // `start_server` takes ownership; the remote-control service needs the
    // same manager for its long-lived session and the config store.
    let remote_session_manager = session_manager.clone();

    // Remember whether we're in zero-file mode before `port` gets shadowed
    // by `server.port()`.
    let zero_file_mode = managed && port.is_some();
    let mut config = nevoflux_daemon::ServerConfig {
        trace_enabled,
        managed,
        data_dir: Some(data_dir.clone()),
        explicit_port: port,
        ..Default::default()
    };
    if let Some(ps) = port_start {
        config.port_start = ps;
    }
    if let Some(pe) = port_end {
        config.port_end = pe;
    }
    let router = std::sync::Arc::new(nevoflux_daemon::Router::new());

    let mut server = nevoflux_daemon::start_server(config, router, session_manager).await?;
    let port = server.port();

    tracing::info!("Daemon started on port {} (managed={})", port, managed);

    // Headless automation mode (P4): serve the task HTTP API. The queue/router/
    // metrics are real (unit-tested); task *execution* (the browser-driving
    // session-runner leaf) is the browser-gated piece — until it lands, submitted
    // tasks report not-yet-wired. The daemon still routes browser_* tools to a
    // registered browser via the P2 binding once the leaf runs them.
    // E2E bug #2 fix: a browser we spawn launches a native-messaging proxy
    // that, in prod mode, discovers a daemon via `daemon-managed.port`. Write
    // it with THIS daemon's port so the proxy connects BACK to us instead of
    // spawning its own managed daemon (which would hold the browser in a
    // different daemon than the one that launched it). Both browser-spawning
    // modes need this, so it sits outside either one.
    if headless || remote_control {
        let managed_port_file = data_dir.join("daemon-managed.port");
        if let Err(e) = std::fs::write(&managed_port_file, port.to_string()) {
            tracing::warn!("failed to write daemon-managed.port: {}", e);
        } else {
            tracing::info!("wrote daemon-managed.port={} for proxy connect-back", port);
        }
    }

    if headless {
        if http_addr.is_some() || openai_addr.is_some() || mcp_addr.is_some() || acp_addr.is_some()
        {
            use nevoflux_daemon::http;
            use std::sync::Arc;
            let metrics = Arc::new(http::metrics::Metrics::default());
            // Real runner: clone profile → spawn browser → bind → run agent →
            // drain, with taint-gated retry. Falls back to a stub if the daemon
            // context or NEVOFLUX_BROWSER_BIN isn't available yet.
            let runner: http::queue::Runner = nevoflux_daemon::automation::build_headless_runner(
                metrics.clone(),
            )
            .unwrap_or_else(|| {
                tracing::warn!(
                    "headless runner context not ready (set NEVOFLUX_BROWSER_BIN); serving stub"
                );
                Arc::new(|id, _req, _sink, _cancel| {
                    Box::pin(async move {
                        http::types::TaskResponse {
                            id,
                            status: http::types::TaskStatus::Failed,
                            attempts: 1,
                            output: None,
                            error: Some("headless runner context unavailable".into()),
                            artifacts: Vec::new(),
                        }
                    })
                })
            });
            let state = http::router::AppState {
                queue: Arc::new(http::queue::TaskQueue::new(runner)),
                metrics,
            };

            // Task API (+ OpenAI merged) on --http-addr; dedicated OpenAI / MCP /
            // ACP ports when their addr is set. All share the same task state.
            if let Some(addr) = http_addr {
                let app = http::router::router(state.clone());
                tokio::spawn(async move {
                    tracing::info!("Headless task API (+ OpenAI /v1) listening on {}", addr);
                    if let Err(e) = http::router::serve(addr, app).await {
                        tracing::error!("task API server error: {}", e);
                    }
                });
            }
            if let Some(addr) = openai_addr {
                let app = http::router::openai_routes().with_state(state.clone());
                tokio::spawn(async move {
                    tracing::info!(
                        "OpenAI-compatible API listening on {} (/v1/chat/completions)",
                        addr
                    );
                    if let Err(e) = http::router::serve(addr, app).await {
                        tracing::error!("OpenAI API server error: {}", e);
                    }
                });
            }
            if let Some(addr) = mcp_addr {
                let app = http::rpc::mcp_routes().with_state(state.clone());
                tokio::spawn(async move {
                    tracing::info!("MCP-over-HTTP listening on {} (POST /mcp)", addr);
                    if let Err(e) = http::router::serve(addr, app).await {
                        tracing::error!("MCP server error: {}", e);
                    }
                });
            }
            if let Some(addr) = acp_addr {
                let app = http::rpc::acp_routes().with_state(state.clone());
                tokio::spawn(async move {
                    tracing::info!("ACP-over-HTTP listening on {} (POST /acp)", addr);
                    if let Err(e) = http::router::serve(addr, app).await {
                        tracing::error!("ACP server error: {}", e);
                    }
                });
            }
        } else if !remote_control {
            // The warning was right when serving an interface was the only
            // reason to run headless. With --remote-control the absence of one
            // is the design, not an oversight.
            tracing::warn!(
                "--headless without --http-addr/--openai-addr/--mcp-addr/--acp-addr: no API served"
            );
        }
    }

    // Remote-control service: a head with nobody in front of it. Started after
    // the server is up (it needs the gateway registry and the message sender),
    // and fatal on any startup failure — a container that half-starts reads as
    // healthy to an orchestrator and is useless to whoever holds the phone.
    if remote_control {
        let Some(browser_bin) = std::env::var_os("NEVOFLUX_BROWSER_BIN").map(PathBuf::from) else {
            eprintln!(
                "--remote-control requires NEVOFLUX_BROWSER_BIN (the path to the browser binary)"
            );
            std::process::exit(2);
        };
        let Some((registry, msg_tx)) = server.remote_wiring() else {
            eprintln!("--remote-control: this daemon exposed no remote-gateway wiring");
            std::process::exit(2);
        };
        let Some(browsers) = nevoflux_daemon::registry::CURRENT_BROWSER_REGISTRY
            .get()
            .cloned()
        else {
            eprintln!("--remote-control: the browser registry was not initialised");
            std::process::exit(2);
        };

        // The chat path reads these bindings through the same global, which is
        // why it stays unset — and the lookup skipped — on the desktop.
        let bindings =
            std::sync::Arc::new(nevoflux_daemon::registry::SessionBrowserBindings::new());
        let _ = nevoflux_daemon::registry::CURRENT_SESSION_BINDINGS.set(bindings.clone());

        let deps = nevoflux_daemon::remote::control_service::ServiceDeps {
            data_dir: data_dir.clone(),
            browser_bin,
            display: std::env::var("DISPLAY").ok(),
            portal_base: std::env::var("NEVOFLUX_PORTAL_URL")
                .unwrap_or_else(|_| "https://portal.nevoflux.app".to_string()),
            registry,
            msg_tx,
            browsers,
            bindings,
            session_manager: remote_session_manager,
        };
        tokio::spawn(async move {
            if let Err(e) = nevoflux_daemon::remote::control_service::run(deps).await {
                eprintln!("remote-control: {e}");
                // This service is the point of the process; if it cannot run,
                // neither can the container.
                std::process::exit(1);
            }
        });
    }

    // Wait for a shutdown signal: Ctrl+C, or the server terminating on its
    // own (managed-mode idle timeout). Waiting only on Ctrl+C left idle
    // managed daemons alive forever: the idle check stopped the accept loop
    // but nothing ever ended the process, so each browser session leaked an
    // orphan daemon holding a port from the 19501-19600 range.
    tokio::select! {
        result = tokio::signal::ctrl_c() => result?,
        _ = server.wait_terminated() => {
            tracing::info!("server terminated on its own (idle timeout); exiting");
        }
    }

    logging::log_shutdown();

    // Arm an independent OS-thread backstop FIRST: from this point the
    // process is guaranteed to die within the grace window no matter what
    // the async path below does. This is what makes Ctrl+C reliable — even
    // if graceful shutdown wedges or the final force-exit deadlocks, the
    // watchdog fires and terminates us.
    spawn_force_exit_watchdog(std::time::Duration::from_secs(8));

    // Gracefully shut down the server (stops listeners and background
    // tasks). Best-effort, and also escapable with a second Ctrl+C, which
    // breaks the select and drops us straight to the force-exit below.
    // (`tokio::signal::ctrl_c()` resolves once per await, so a fresh await
    // here is what makes a *second* Ctrl+C effective.)
    let graceful = server.shutdown();
    tokio::pin!(graceful);
    tokio::select! {
        _ = &mut graceful => {
            tracing::info!("graceful shutdown complete");
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::warn!("second Ctrl+C received during shutdown; forcing exit");
        }
    }

    // Cleanup — only remove files we actually wrote.
    // In managed+port mode no files were written, so nothing to clean up.
    if !zero_file_mode {
        let data_dir = get_data_dir();
        let (port_name, pid_name) = if managed {
            ("daemon-managed.port", "daemon-managed.pid")
        } else {
            ("daemon.port", "daemon.pid")
        };
        let _ = std::fs::remove_file(data_dir.join(port_name));
        let _ = std::fs::remove_file(data_dir.join(pid_name));
    }

    // Force-exit the process. Loop iterations run Agent::run() inside
    // spawn_blocking — those threads cannot be interrupted by tokio
    // cancellation, so the tokio runtime drop would otherwise hang waiting
    // for the blocking pool to drain. `force_exit` uses TerminateProcess on
    // Windows (see its doc) rather than `std::process::exit`, whose stdout
    // flush + ExitProcess loader-lock teardown can itself deadlock here.
    force_exit(0);
}

/// Get the database path for storage.
fn get_db_path() -> PathBuf {
    get_data_dir().join("nevoflux.db")
}

/// Open storage, creating if necessary.
fn open_storage() -> Result<Storage, Box<dyn std::error::Error>> {
    let db_path = get_db_path();
    ensure_data_dir()?;
    Ok(Storage::open(&db_path)?)
}

/// Run the config show command.
fn run_config_show() {
    match open_storage() {
        Ok(storage) => match storage.config().list() {
            Ok(entries) => {
                if entries.is_empty() {
                    println!("No configuration entries found.");
                    println!(
                        "Run 'nevoflux-agent config init' to initialize default configuration."
                    );
                } else {
                    println!("Configuration:");
                    println!("{}", "-".repeat(60));
                    for entry in entries {
                        let value_str = serde_json::to_string_pretty(&entry.value)
                            .unwrap_or_else(|_| entry.value.to_string());
                        println!("{} = {}", entry.key, value_str);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error listing configuration: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error opening storage: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run the config init command.
fn run_config_init() {
    match open_storage() {
        Ok(storage) => {
            let defaults = [
                ("app.name", serde_json::json!("NevoFlux")),
                ("app.version", serde_json::json!(env!("CARGO_PKG_VERSION"))),
                ("daemon.port", serde_json::json!(0)), // 0 means auto-assign
                ("daemon.auto_start", serde_json::json!(true)),
                ("logging.level", serde_json::json!("info")),
                ("logging.file", serde_json::json!(null)),
            ];

            for (key, value) in defaults {
                // Only set if not already exists
                match storage.config().get(key) {
                    Ok(Some(_)) => {
                        println!("  {} (already set, skipping)", key);
                    }
                    Ok(None) => {
                        if let Err(e) = storage.config().set(key, value.clone()) {
                            eprintln!("Error setting {}: {}", key, e);
                        } else {
                            println!("  {} = {}", key, value);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error checking {}: {}", key, e);
                    }
                }
            }
            println!("\nConfiguration initialized.");
        }
        Err(e) => {
            eprintln!("Error opening storage: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run the config get command.
fn run_config_get(key: &str) {
    match open_storage() {
        Ok(storage) => match storage.config().get(key) {
            Ok(Some(value)) => {
                let value_str =
                    serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
                println!("{}", value_str);
            }
            Ok(None) => {
                eprintln!("Key '{}' not found", key);
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error getting configuration: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error opening storage: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run the config set command.
fn run_config_set(key: &str, value: &str) {
    match open_storage() {
        Ok(storage) => {
            // Try to parse as JSON, fall back to string
            let json_value: serde_json::Value = serde_json::from_str(value)
                .unwrap_or_else(|_| serde_json::Value::String(value.to_string()));

            match storage.config().set(key, json_value.clone()) {
                Ok(()) => {
                    println!("Set {} = {}", key, json_value);
                }
                Err(e) => {
                    eprintln!("Error setting configuration: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Error opening storage: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run the config delete command.
fn run_config_delete(key: &str) {
    match open_storage() {
        Ok(storage) => match storage.config().delete(key) {
            Ok(true) => {
                println!("Deleted '{}'", key);
            }
            Ok(false) => {
                eprintln!("Key '{}' not found", key);
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error deleting configuration: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error opening storage: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run the config list command.
fn run_config_list(prefix: &str) {
    match open_storage() {
        Ok(storage) => {
            let result = if prefix.is_empty() {
                storage.config().list()
            } else {
                storage.config().list_by_prefix(prefix)
            };

            match result {
                Ok(entries) => {
                    if entries.is_empty() {
                        if prefix.is_empty() {
                            println!("No configuration entries found.");
                        } else {
                            println!("No configuration entries found with prefix '{}'", prefix);
                        }
                    } else {
                        for entry in entries {
                            let value_str = serde_json::to_string(&entry.value)
                                .unwrap_or_else(|_| entry.value.to_string());
                            println!("{} = {}", entry.key, value_str);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error listing configuration: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Error opening storage: {}", e);
            std::process::exit(1);
        }
    }
}

/// Run the setup wizard.
fn run_setup() {
    println!("NevoFlux Setup Wizard");
    println!("{}", "=".repeat(40));
    println!();

    // Initialize default configuration
    println!("Initializing default configuration...");
    run_config_init();

    println!();
    println!("Setup complete!");
    println!();
    println!("Next steps:");
    println!("  1. Start the daemon: nevoflux --daemon");
    println!("  2. Check status: nevoflux --status");
    println!("  3. View configuration: nevoflux config show");
}

/// Handle config subcommands.
fn handle_config_command(action: ConfigAction) {
    match action {
        ConfigAction::Show => run_config_show(),
        ConfigAction::Init => run_config_init(),
        ConfigAction::Get { key } => run_config_get(&key),
        ConfigAction::Set { key, value } => run_config_set(&key, &value),
        ConfigAction::Delete { key } => run_config_delete(&key),
        ConfigAction::List { prefix } => run_config_list(&prefix),
    }
}

/// Connect to whichever daemon is running: a manually-started dev daemon
/// (daemon.port) or a proxy-managed one (daemon-managed.port). Resolves the
/// data dir the same way the daemon does so NEVOFLUX_DATA_DIR is honored.
async fn connect_pack_client() -> Result<nevoflux_bridge::DaemonClient, Box<dyn std::error::Error>>
{
    use nevoflux_bridge::{BridgeConfig, ConnectionMode, DaemonClient};

    let data_dir = get_data_dir();
    let mut connected = None;
    let mut last_err = String::new();
    for mode in [ConnectionMode::Dev, ConnectionMode::Prod] {
        let cfg = BridgeConfig::new()
            .with_mode(mode)
            .with_data_dir(data_dir.clone())
            .with_auto_launch(false);
        let mut c = DaemonClient::new("pack-cli", cfg);
        match c.connect().await {
            Ok(()) => {
                connected = Some(c);
                break;
            }
            Err(e) => last_err = e.to_string(),
        }
    }
    connected.ok_or_else(|| {
        format!(
            "cannot reach daemon (is it running? start it with `nevoflux --daemon`): {last_err}"
        )
        .into()
    })
}

/// Send one `pack.*` command on a fresh connection and return the unwrapped
/// `data` from the matching `system_response` (or an error).
async fn send_pack_rpc(
    command: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let mut client = connect_pack_client().await?;

    let request_id = format!("pack-cli-{command}");
    let payload = serde_json::json!({
        "type": "system_command",
        "payload": {
            "command": command,
            "request_id": request_id,
            "params": params,
        }
    });
    client.send_chat(&request_id, payload).await?;

    // Read responses until we see the matching system_response.
    loop {
        let env = client.recv().await?;
        let p = &env.payload;
        if p.get("type").and_then(|t| t.as_str()) == Some("system_response")
            && p.get("payload")
                .and_then(|x| x.get("command"))
                .and_then(|c| c.as_str())
                == Some(command)
        {
            let inner = &p["payload"];
            if inner["success"].as_bool().unwrap_or(false) {
                return Ok(inner["data"].clone());
            }
            let msg = inner["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            return Err(format!("{command} failed: {msg}").into());
        }
    }
}

/// True for sources the daemon must fetch over the network.
fn is_remote_source(source: &str) -> bool {
    source.starts_with("github:") || source.starts_with("https://github.com/")
}

/// Handle `nevoflux pack <action>` by issuing a `pack.*` RPC to the daemon.
///
/// Uses the same length-prefixed JSON transport (`DaemonClient`) the daemon
/// speaks. The CLI uses `wait:true` so install/uninstall/update block until the
/// lifecycle finishes and the final result is returned inline.
async fn handle_pack_command(action: PackAction) -> Result<(), Box<dyn std::error::Error>> {
    // Remote install: preview + confirm unless --yes.
    if let PackAction::Install { source, yes, .. } = &action {
        if is_remote_source(source) && !yes {
            // Inspect first so the user sees what they're about to trust.
            let preview =
                send_pack_rpc("pack.inspect", serde_json::json!({ "source": source })).await?;
            println!("{}", serde_json::to_string_pretty(&preview).unwrap());
            eprintln!(
                "\n⚠  Unreviewed source. Installing trusts the author: skills can direct the agent and canvas-tools can run commands on your machine."
            );
            eprint!("Proceed with install? [y/N] ");
            use std::io::Write as _;
            std::io::stderr().flush().ok();
            use std::io::BufRead as _;
            let mut line = String::new();
            std::io::stdin().lock().read_line(&mut line)?;
            if !matches!(line.trim(), "y" | "Y" | "yes") {
                eprintln!("Aborted.");
                return Ok(());
            }
        }
    }

    // Build the (command, params) pair. Per-command args go inside `params`,
    // which the daemon's handler reads flat.
    let (command, params): (&str, serde_json::Value) = match &action {
        PackAction::Validate { source } => {
            ("pack.validate", serde_json::json!({ "source": source }))
        }
        PackAction::Inspect { source } => ("pack.inspect", serde_json::json!({ "source": source })),
        PackAction::Install { source, force, .. } => (
            "pack.install",
            serde_json::json!({ "source": source, "force": force, "wait": true }),
        ),
        PackAction::Uninstall {
            name,
            purge_data,
            force,
        } => (
            "pack.uninstall",
            serde_json::json!({ "name": name, "purge_data": purge_data, "force": force }),
        ),
        PackAction::Update { source } => ("pack.update", serde_json::json!({ "source": source })),
        PackAction::List => ("pack.list", serde_json::json!({})),
        PackAction::Status { name } => ("pack.status", serde_json::json!({ "name": name })),
    };

    let data = send_pack_rpc(command, params).await?;
    println!("{}", serde_json::to_string_pretty(&data).unwrap());
    Ok(())
}

/// Handle `nevoflux account <action>`.
///
/// `login` runs the RFC 8628 device grant against nevoflux.app entirely
/// in-process — no daemon needed — so a container can mint its own account token
/// before `--remote-control` starts, instead of copying one off a desktop. All
/// human-facing prompts and status go to **stderr** so that, with `--print-token`,
/// stdout carries only the token and can be piped straight into a secret.
async fn handle_account_command(action: AccountAction) -> Result<(), Box<dyn std::error::Error>> {
    use nevoflux_daemon::remote::account::{self, FileTokenStore, PollOutcome, TokenStore};
    use nevoflux_daemon::remote::start::{account_token_path, SERVICE_TOKEN_ENV};

    // The env override is the same credential the file holds; report it so a
    // `status`/`logout` is not misread when a container also sets the env.
    let env_token = std::env::var(SERVICE_TOKEN_ENV)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    match action {
        AccountAction::Status => {
            let path = account_token_path();
            let file_token = FileTokenStore::new(path.clone())
                .load()
                .ok()
                .flatten()
                .filter(|s| !s.trim().is_empty());
            if env_token.is_some() {
                println!("logged in (via {SERVICE_TOKEN_ENV}; overrides any file)");
            } else if file_token.is_some() {
                println!("logged in (token at {})", path.display());
            } else {
                println!("not logged in");
            }
            Ok(())
        }
        AccountAction::Logout => {
            let path = account_token_path();
            FileTokenStore::new(path.clone()).clear()?;
            eprintln!("Removed {}", path.display());
            if env_token.is_some() {
                eprintln!(
                    "Note: {SERVICE_TOKEN_ENV} is still set in this environment and takes \
                     precedence over the (now-removed) file."
                );
            }
            Ok(())
        }
        AccountAction::Login {
            print_token,
            no_save,
        } => {
            let base = std::env::var("NEVOFLUX_ACCOUNT_URL")
                .unwrap_or_else(|_| "https://nevoflux.app".to_string());
            // Same client id the daemon's device-grant RPC uses, so the auth
            // server treats a CLI login and a sidebar login identically.
            const CLIENT_ID: &str = "nevoflux-daemon";

            // 1) Start the grant and show the user what to approve.
            let dc = account::request_device_code(&base, CLIENT_ID).await?;
            eprintln!("\nTo sign in, open this URL on any device and enter the code:\n");
            eprintln!("    {}", dc.verification_uri);
            eprintln!("\n    code:  {}\n", dc.user_code);
            eprintln!(
                "Waiting for approval (the code expires in {}s)...",
                dc.expires_in_secs
            );

            // 2) Poll until approved / denied / expired. A transient poll error
            //    is retried rather than fatal — the window is minutes long and a
            //    blip should not throw away an in-progress sign-in.
            let mut interval = std::time::Duration::from_secs(dc.interval_secs.max(1));
            let deadline = std::time::Instant::now()
                + std::time::Duration::from_secs(dc.expires_in_secs.max(1));
            let token = loop {
                if std::time::Instant::now() >= deadline {
                    return Err("device code expired before it was approved".into());
                }
                tokio::time::sleep(interval).await;
                match account::poll_device_token(&base, CLIENT_ID, &dc.device_code).await {
                    Ok(PollOutcome::Pending) => {}
                    Ok(PollOutcome::SlowDown) => {
                        interval += std::time::Duration::from_secs(5);
                    }
                    Ok(PollOutcome::Denied(d)) => {
                        return Err(format!("sign-in failed: {d}").into());
                    }
                    Ok(PollOutcome::Token(t)) => break t,
                    Err(e) => eprintln!("(poll error, retrying: {e})"),
                }
            };

            // 3) Persist unless told not to. 0600 on Unix — this is account-
            //    bearer material, not world-readable config.
            if no_save {
                eprintln!("Signed in. Token NOT saved (--no-save).");
            } else {
                let path = account_token_path();
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir).ok();
                }
                FileTokenStore::new(path.clone()).save(&token)?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
                }
                eprintln!("Signed in. Token saved to {}", path.display());
                if env_token.is_some() {
                    eprintln!(
                        "Note: {SERVICE_TOKEN_ENV} is set and overrides this file at runtime."
                    );
                }
            }

            // 4) Emit the token on stdout when asked (forced by --no-save, or the
            //    token would be lost). This is the only thing on stdout.
            if print_token || no_save {
                println!("{token}");
            }
            Ok(())
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Handle subcommands first (they don't require async)
    if let Some(command) = cli.command {
        match command {
            Commands::Config { action } => {
                handle_config_command(action);
                return;
            }
            Commands::Pack { action } => {
                if let Err(e) = handle_pack_command(action).await {
                    eprintln!("pack: {e}");
                    std::process::exit(1);
                }
                return;
            }
            Commands::Account { action } => {
                if let Err(e) = handle_account_command(action).await {
                    eprintln!("account: {e}");
                    std::process::exit(1);
                }
                return;
            }
            Commands::Setup => {
                run_setup();
                return;
            }
            Commands::Completions { shell } => {
                completions::generate_completions(shell);
                return;
            }
            Commands::External(_) => {
                // Firefox passes manifest path and extension ID as arguments
                // Ignore them and continue to proxy mode (default behavior)
            }
        }
    }

    // Handle flags
    let dev_mode = cli.dev
        || std::env::var("NEVOFLUX_DEV")
            .map(|v| v == "1")
            .unwrap_or(false);

    // A dev daemon usually learns its dev-ness from the `--dev` flag, which does
    // NOT put NEVOFLUX_DEV in the process env. Browsers we spawn for headless
    // runs inherit our env, and their native-messaging proxy selects Dev vs Prod
    // solely from NEVOFLUX_DEV (see run_proxy / dev_mode above). Without this,
    // a spawned headless browser's proxy defaults to Prod, launches its own
    // managed daemon on 19501+, and never registers with THIS dev daemon on
    // 19500 -> the headless readiness barrier times out. Re-export the signal so
    // it propagates to every child process we spawn.
    if dev_mode {
        std::env::set_var("NEVOFLUX_DEV", "1");
    }

    if cli.daemon {
        if let Err(e) = run_daemon(
            cli.verbose,
            cli.trace,
            cli.port_start,
            cli.port_end,
            cli.port,
            cli.managed,
            cli.headless,
            cli.remote_control,
            cli.http_addr,
            cli.openai_addr,
            cli.mcp_addr,
            cli.acp_addr,
        )
        .await
        {
            eprintln!("Daemon error: {}", e);
            std::process::exit(1);
        }
    } else if cli.mcp {
        if let Err(e) = run_mcp(cli.verbose).await {
            eprintln!("MCP server error: {}", e);
            std::process::exit(1);
        }
    } else if cli.status {
        run_status();
    } else if cli.stop {
        run_stop();
    } else if let Err(e) = run_proxy(cli.verbose, dev_mode).await {
        eprintln!("Proxy error: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Mutex to serialize tests that modify NEVOFLUX_DATA_DIR environment variable.
    // This prevents race conditions when tests run in parallel.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_get_data_dir_returns_path() {
        let dir = get_data_dir();
        assert!(dir.exists() || dir.parent().map(|p| p.exists()).unwrap_or(false));
    }

    #[test]
    fn test_port_file_path() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Use temp dir that contains "nevoflux" in path name for test
        let temp = tempfile::Builder::new()
            .prefix("nevoflux-test")
            .tempdir()
            .unwrap();
        std::env::set_var("NEVOFLUX_DATA_DIR", temp.path());

        let dir = get_data_dir();
        let port_file = dir.join("daemon.port");
        assert!(port_file.to_string_lossy().contains("nevoflux"));

        std::env::remove_var("NEVOFLUX_DATA_DIR");
    }

    #[test]
    fn test_check_daemon_status_not_running() {
        let _guard = ENV_MUTEX.lock().unwrap();

        let temp = tempfile::TempDir::new().unwrap();
        std::env::set_var("NEVOFLUX_DATA_DIR", temp.path());

        let status = check_daemon_status();
        assert!(matches!(status, DaemonStatus::NotRunning));

        std::env::remove_var("NEVOFLUX_DATA_DIR");
    }

    #[test]
    fn test_check_daemon_status_with_port_file() {
        let _guard = ENV_MUTEX.lock().unwrap();

        let temp = tempfile::TempDir::new().unwrap();
        std::fs::write(temp.path().join("daemon.port"), "19500").unwrap();
        std::fs::write(temp.path().join("daemon.pid"), "12345").unwrap();
        std::env::set_var("NEVOFLUX_DATA_DIR", temp.path());

        let status = check_daemon_status();
        // Should be Stale since PID 12345 is not running
        assert!(matches!(status, DaemonStatus::Stale { .. }));

        std::env::remove_var("NEVOFLUX_DATA_DIR");
    }

    #[test]
    fn test_stop_daemon_no_files() {
        let _guard = ENV_MUTEX.lock().unwrap();

        let temp = tempfile::TempDir::new().unwrap();
        std::env::set_var("NEVOFLUX_DATA_DIR", temp.path());

        let result = stop_daemon();
        assert!(result.is_ok());

        std::env::remove_var("NEVOFLUX_DATA_DIR");
    }

    #[test]
    fn test_stop_daemon_cleans_files() {
        let _guard = ENV_MUTEX.lock().unwrap();

        let temp = tempfile::TempDir::new().unwrap();
        let port_file = temp.path().join("daemon.port");
        let pid_file = temp.path().join("daemon.pid");
        std::fs::write(&port_file, "19500").unwrap();
        std::fs::write(&pid_file, "99999").unwrap(); // Non-existent PID

        std::env::set_var("NEVOFLUX_DATA_DIR", temp.path());

        let result = stop_daemon();
        assert!(result.is_ok());
        assert!(!port_file.exists());
        assert!(!pid_file.exists());

        std::env::remove_var("NEVOFLUX_DATA_DIR");
    }

    #[test]
    fn test_acquire_daemon_lock_succeeds() {
        let temp = tempfile::TempDir::new().unwrap();
        std::env::set_var("NEVOFLUX_DATA_DIR", temp.path());

        let lock = acquire_daemon_lock(false);
        assert!(lock.is_ok());

        std::env::remove_var("NEVOFLUX_DATA_DIR");
    }
}
