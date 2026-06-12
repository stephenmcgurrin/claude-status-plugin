use std::collections::HashSet;
use std::env;
use std::ffi::CString;
use std::fs;
use std::io::{self, BufRead, BufReader, IsTerminal, Write};
use std::os::unix::process::{parent_id, CommandExt};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::thread;
use std::time::Duration;

use serde_json::Value;

unsafe extern "C" {
    fn notify_post(name: *const std::ffi::c_char) -> u32;
}

const NOTIFICATION_NAME: &str = "com.poisonpenllc.Claude-Status.session-changed";
const POLL_INTERVAL: Duration = Duration::from_millis(100);

// ---------------------------------------------------------------------------
// Shared utilities (matching set-session-name)
// ---------------------------------------------------------------------------

fn post_darwin_notification() {
    let name = CString::new(NOTIFICATION_NAME).expect("invalid notification name");
    unsafe {
        notify_post(name.as_ptr());
    }
}

fn get_ppid_of(pid: u32) -> Option<u32> {
    const PROC_PIDTBSDINFO: libc::c_int = 3;
    const PROC_PIDTBSDINFO_SIZE: libc::c_int = 136;

    let mut info = vec![0u8; PROC_PIDTBSDINFO_SIZE as usize];

    unsafe extern "C" {
        fn proc_pidinfo(
            pid: libc::c_int,
            flavor: libc::c_int,
            arg: u64,
            buffer: *mut libc::c_void,
            buffersize: libc::c_int,
        ) -> libc::c_int;
    }

    let ret = unsafe {
        proc_pidinfo(
            pid as libc::c_int,
            PROC_PIDTBSDINFO,
            0,
            info.as_mut_ptr() as *mut libc::c_void,
            PROC_PIDTBSDINFO_SIZE,
        )
    };

    if ret <= 0 {
        return None;
    }

    let ppid = u32::from_ne_bytes([info[24], info[25], info[26], info[27]]);
    if ppid == 0 { None } else { Some(ppid) }
}

fn write_atomic(path: &Path, data: &[u8]) -> io::Result<()> {
    let dir = path
        .parent()
        .expect("cstatus file must have a parent directory");
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    tmp.write_all(data)?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

fn utc_timestamp() -> String {
    let mut tv = libc::timeval {
        tv_sec: 0,
        tv_usec: 0,
    };
    unsafe {
        libc::gettimeofday(&mut tv, std::ptr::null_mut());
    }
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe {
        libc::gmtime_r(&tv.tv_sec, &mut tm);
    }
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
    )
}

fn pid_is_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

fn transcript_sibling(transcript_path: &str, ext: &str) -> PathBuf {
    Path::new(transcript_path).with_extension(ext)
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum SessionState {
    Active,
    Waiting,
    Idle,
    Compacting,
}

impl SessionState {
    fn as_str(&self) -> &'static str {
        match self {
            SessionState::Active => "active",
            SessionState::Waiting => "waiting",
            SessionState::Idle => "idle",
            SessionState::Compacting => "compacting",
        }
    }
}

#[derive(Debug, Clone)]
struct StatusInfo {
    session_id: String,
    pid: u32,
    ppid: u32,
    state: SessionState,
    activity: String,
    cwd: String,
    event: String,
    session_name: Option<String>,
}

impl StatusInfo {
    fn to_json(&self) -> String {
        let mut obj = serde_json::json!({
            "session_id": self.session_id,
            "pid": self.pid,
            "ppid": self.ppid,
            "state": self.state.as_str(),
            "activity": self.activity,
            "timestamp": utc_timestamp(),
            "cwd": self.cwd,
            "event": self.event,
        });
        if let Some(ref name) = self.session_name {
            obj["session_name"] = Value::String(name.clone());
        }
        let mut s = serde_json::to_string(&obj).expect("serialize status");
        s.push('\n');
        s
    }
}

// ---------------------------------------------------------------------------
// JSONL-driven state machine
// ---------------------------------------------------------------------------

struct DaemonState {
    state: SessionState,
    activity: String,
    event: String,
    active_agents: HashSet<String>,
    session_name: Option<String>,
    /// Set when compact_boundary fires. Suppresses state changes from replayed
    /// context (user messages, progress) until the next real assistant response.
    compacting: bool,
}

impl DaemonState {
    fn new() -> Self {
        Self {
            state: SessionState::Idle,
            activity: String::new(),
            event: String::new(),
            active_agents: HashSet::new(),
            session_name: None,
            compacting: false,
        }
    }

    /// Process a JSONL line and return true if state changed.
    fn process_line(&mut self, line: &str) -> bool {
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => return false,
        };

        let line_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");

        let old_state = self.state.clone();
        let old_activity = self.activity.clone();

        // Skip meta messages (local command output, plugin commands, etc.)
        // These are not real user/assistant turns and should not affect state.
        if v.get("isMeta").and_then(|m| m.as_bool()).unwrap_or(false) {
            return false;
        }

        match line_type {
            "assistant" => self.process_assistant(&v),
            "user" => {
                if self.compacting {
                    // Suppress replayed context during compaction
                    self.event = "user".to_string();
                } else {
                    self.process_user(&v);
                }
            }
            "progress" => {
                if self.compacting {
                    self.event = "progress".to_string();
                } else {
                    self.process_progress(&v);
                }
            }
            "system" => self.process_system(&v),
            // No-op types: file-history-snapshot, last-prompt, pr-link, queue-operation
            _ => {}
        }

        // Compaction detection via agentId for long compactions that spawn
        // a compact agent (agentId prefixed with "acompact-")
        if !self.compacting && self.state != SessionState::Idle {
            if let Some(agent_id) = v.get("agentId").and_then(|a| a.as_str()) {
                if agent_id.starts_with("acompact-") {
                    self.compacting = true;
                    self.state = SessionState::Compacting;
                    if !self.activity.is_empty() {
                        self.activity = format!("compacting ({})", self.activity);
                    } else {
                        self.activity = "compacting".to_string();
                    }
                }
            }
        }

        self.state != old_state || self.activity != old_activity
    }

    fn process_assistant(&mut self, v: &Value) {
        // An assistant response after compaction means the replayed context is
        // done and the model is responding to the post-compaction prompt.
        self.compacting = false;

        let message = match v.get("message") {
            Some(m) => m,
            None => return,
        };

        let stop_reason = message
            .get("stop_reason")
            .and_then(|s| s.as_str())
            .unwrap_or("");

        let empty_content = Vec::new();
        let content = message
            .get("content")
            .and_then(|c| c.as_array())
            .unwrap_or(&empty_content);

        // Track agent spawns — look for tool_use blocks with name "Agent"
        for block in content {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if block_type == "tool_use" {
                let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                if name == "Agent"
                    && let Some(id) = block.get("id").and_then(|i| i.as_str())
                {
                    self.active_agents.insert(id.to_string());
                }
            }
        }

        self.event = "assistant".to_string();

        match stop_reason {
            "" => {
                // stop_reason is null (streaming) — as_str() returns None, unwrap_or("")
                if message.get("stop_reason").is_none_or(|s| s.is_null()) {
                    self.state = SessionState::Active;
                    self.activity = String::new();
                }
            }
            "tool_use" => {
                let tool_name = content
                    .iter()
                    .rev()
                    .find_map(|block| {
                        let bt = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if bt == "tool_use" {
                            block.get("name").and_then(|n| n.as_str()).map(String::from)
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                self.state = SessionState::Active;
                self.activity = tool_name;
            }
            "end_turn" => {
                if !self.active_agents.is_empty() {
                    // Agents still running — stay active
                    self.state = SessionState::Active;
                    self.activity = "subagent".to_string();
                } else {
                    // Question detection on the message we just parsed
                    if self.detect_question(content) {
                        self.state = SessionState::Waiting;
                        self.activity = "question".to_string();
                    } else {
                        self.state = SessionState::Idle;
                        self.activity = String::new();
                    }
                }
            }
            "stop_sequence" => {
                self.state = SessionState::Idle;
                self.activity = String::new();
            }
            "max_tokens" => {
                // Response hit token limit — turn is over, go idle
                self.state = SessionState::Idle;
                self.activity = String::new();
            }
            _ => {
                // Unknown stop_reason — don't change state
            }
        }
    }

    fn process_user(&mut self, v: &Value) {
        let message = match v.get("message") {
            Some(m) => m,
            None => return,
        };

        let empty_content = Vec::new();
        let content = message
            .get("content")
            .and_then(|c| c.as_array())
            .unwrap_or(&empty_content);

        // Check if this is an async agent launch (tool_result returned immediately
        // but agent is still running in the background)
        let is_async = v
            .get("toolUseResult")
            .and_then(|r| r.get("isAsync"))
            .and_then(|a| a.as_bool())
            .unwrap_or(false);

        // Check for tool_result blocks — may complete an agent
        let mut has_tool_result = false;
        let mut has_text = false;
        for block in content {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match block_type {
                "tool_result" => {
                    has_tool_result = true;
                    if !is_async {
                        if let Some(tool_use_id) =
                            block.get("tool_use_id").and_then(|i| i.as_str())
                        {
                            self.active_agents.remove(tool_use_id);
                        }
                    }
                }
                "text" => {
                    let text = block.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        has_text = true;
                    }
                }
                "image" => {
                    has_text = true;
                }
                _ => {}
            }
        }

        // Also handle content as string (user:text type has content as array with text blocks,
        // but some entries have content as a string at the message level)
        if !has_tool_result
            && !has_text
            && let Some(content_str) = message.get("content").and_then(|c| c.as_str())
            && !content_str.is_empty()
        {
            has_text = true;
        }

        self.event = "user".to_string();

        // Only transition to "thinking" for real user prompts (which have promptId).
        // Local command output (e.g. /plugin, /reload-plugins) generates user messages
        // without promptId that should not change state.
        let is_real_prompt = v.get("promptId").is_some_and(|p| !p.is_null());

        if has_text && !has_tool_result && is_real_prompt {
            // A new user prompt means any pending agents from the previous
            // turn were cancelled/interrupted — clear stale tracking.
            self.active_agents.clear();
            self.state = SessionState::Active;
            self.activity = "thinking".to_string();
        }
        // tool_result-only messages don't change state (the assistant response will)
    }

    fn process_progress(&mut self, v: &Value) {
        let data_type = v
            .get("data")
            .and_then(|d| d.get("type"))
            .and_then(|t| t.as_str())
            .unwrap_or("");

        self.event = format!("progress:{}", data_type);

        match data_type {
            "agent_progress" => {
                self.state = SessionState::Active;
                self.activity = "subagent".to_string();
            }
            "bash_progress" => {
                self.state = SessionState::Active;
                self.activity = "bash".to_string();
            }
            "mcp_progress" => {
                self.state = SessionState::Active;
                self.activity = "mcp".to_string();
            }
            // hook_progress, query_update, search_results_received, waiting_for_task — no change
            _ => {}
        }
    }

    fn process_system(&mut self, v: &Value) {
        let subtype = v.get("subtype").and_then(|s| s.as_str()).unwrap_or("");

        self.event = format!("system:{}", subtype);

        match subtype {
            "compact_boundary" => {
                // compact_boundary fires during compaction. Set compacting state
                // and suppress replayed context until the next assistant response.
                self.compacting = true;
                self.state = SessionState::Compacting;
                self.activity = "compacting".to_string();
            }
            "turn_duration" => {
                // turn_duration fires after a turn completes. Any agents
                // still tracked are stale (e.g. user cancelled/interrupted).
                self.active_agents.clear();
                // If we're still in Active state (e.g. the final assistant
                // message had stop_reason: null from streaming, or stale
                // agents kept us active), transition to idle.
                if self.state == SessionState::Active {
                    self.state = SessionState::Idle;
                    self.activity = String::new();
                }
            }
            _ => {}
        }
    }

    /// Detect if the assistant is asking the user a question.
    /// Checks the last paragraph of the last text block for a sentence ending with '?'.
    fn detect_question(&self, content: &[Value]) -> bool {
        for block in content.iter().rev() {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if block_type == "text" {
                let text = block.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if text.is_empty() {
                    continue;
                }
                // Get the last paragraph (after the last blank line)
                let last_paragraph = text
                    .rsplit("\n\n")
                    .next()
                    .unwrap_or(text)
                    .trim();
                return last_paragraph.contains('?');
            }
        }
        false
    }

    /// Process a .csignal file and return true if state changed.
    fn process_signal(&mut self, signal: &Value) -> bool {
        let old_state = self.state.clone();
        let old_activity = self.activity.clone();

        let signal_type = signal.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match signal_type {
            "permission_request" => {
                let tool_name = signal
                    .get("tool_name")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                self.state = SessionState::Waiting;
                self.activity = tool_name;
                self.event = "permission_request".to_string();
            }
            "elicitation_dialog" => {
                self.state = SessionState::Waiting;
                self.activity = String::new();
                self.event = "elicitation_dialog".to_string();
            }
            "idle_prompt" => {
                // idle_prompt means the session is waiting for user input.
                // Transition to idle from active (if no agents) or compacting
                // (post-compact replay finished without an assistant response).
                let should_idle = match self.state {
                    SessionState::Active => self.active_agents.is_empty(),
                    SessionState::Compacting => true,
                    _ => false,
                };
                if should_idle {
                    self.compacting = false;
                    self.state = SessionState::Idle;
                    self.activity = String::new();
                    self.event = "idle_prompt".to_string();
                }
            }
            "pre_compact" => {
                // PreCompact hook fires before compaction starts.
                self.compacting = true;
                self.state = SessionState::Compacting;
                self.activity = "compacting".to_string();
                self.event = "pre_compact".to_string();
            }
            _ => {}
        }

        self.state != old_state || self.activity != old_activity
    }
}

// ---------------------------------------------------------------------------
// Hook mode: SessionStart
// ---------------------------------------------------------------------------

fn hook_session_start(input: &Value) -> Result<(), String> {
    let session_id = input
        .get("session_id")
        .and_then(|s| s.as_str())
        .ok_or("missing session_id")?;
    let transcript_path = input
        .get("transcript_path")
        .and_then(|s| s.as_str())
        .ok_or("missing transcript_path")?;
    let cwd = input
        .get("cwd")
        .and_then(|s| s.as_str())
        .ok_or("missing cwd")?;

    let pid: u32 = env::var("CLAUDE_PID")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(parent_id);
    let ppid = get_ppid_of(pid).unwrap_or(0);

    let cstatus = transcript_sibling(transcript_path, "cstatus");

    // Ensure the project directory exists (may be missing on first run in a new repo)
    if let Some(dir) = cstatus.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("create project dir: {}", e))?;
    }

    // Write initial .cstatus
    let status = StatusInfo {
        session_id: session_id.to_string(),
        pid,
        ppid,
        state: SessionState::Idle,
        activity: String::new(),
        cwd: cwd.to_string(),
        event: "SessionStart".to_string(),
        session_name: None,
    };
    write_atomic(&cstatus, status.to_json().as_bytes())
        .map_err(|e| format!("write cstatus: {}", e))?;

    // Check if a daemon is already running (e.g. session resume)
    let cpid_path = transcript_sibling(transcript_path, "cpid");
    if let Ok(pid_str) = fs::read_to_string(&cpid_path) {
        if let Ok(daemon_pid) = pid_str.trim().parse::<u32>() {
            if pid_is_alive(daemon_pid) {
                // Daemon already running — just update .cstatus and notify
                post_darwin_notification();
                return Ok(());
            }
        }
    }

    // Spawn daemon — clean up .cstatus if spawn fails
    let exe = env::current_exe().map_err(|e| {
        let _ = fs::remove_file(&cstatus);
        format!("current_exe: {}", e)
    })?;

    // Log daemon stderr to a file for crash debugging
    let log_path = transcript_sibling(transcript_path, "clog");
    let log_file = fs::File::create(&log_path).ok();
    let stderr_cfg = match log_file {
        Some(f) => Stdio::from(f),
        None => Stdio::null(),
    };

    Command::new(exe)
        .arg("--daemon")
        .arg("--transcript-path")
        .arg(transcript_path)
        .arg("--session-id")
        .arg(session_id)
        .arg("--cwd")
        .arg(cwd)
        .arg("--pid")
        .arg(pid.to_string())
        .arg("--ppid")
        .arg(ppid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr_cfg)
        .process_group(0) // New process group so hook runner can't kill daemon
        .spawn()
        .map_err(|e| {
            let _ = fs::remove_file(&cstatus);
            format!("spawn daemon: {}", e)
        })?;

    post_darwin_notification();
    Ok(())
}

// ---------------------------------------------------------------------------
// Hook mode: SessionEnd (safety-net)
// ---------------------------------------------------------------------------

fn hook_session_end(input: &Value) -> Result<(), String> {
    let transcript_path = input
        .get("transcript_path")
        .and_then(|s| s.as_str())
        .ok_or("missing transcript_path")?;

    let cstatus = transcript_sibling(transcript_path, "cstatus");
    let csignal = transcript_sibling(transcript_path, "csignal");
    let cpid = transcript_sibling(transcript_path, "cpid");
    let clog = transcript_sibling(transcript_path, "clog");

    let _ = fs::remove_file(&cstatus);
    let _ = fs::remove_file(&csignal);
    let _ = fs::remove_file(&cpid);
    let _ = fs::remove_file(&clog);

    post_darwin_notification();
    Ok(())
}

// ---------------------------------------------------------------------------
// Signal mode
// ---------------------------------------------------------------------------

fn signal_mode(input: &Value) -> Result<(), String> {
    let transcript_path = input
        .get("transcript_path")
        .and_then(|s| s.as_str())
        .ok_or("missing transcript_path")?;
    let hook_event = input
        .get("hook_event_name")
        .and_then(|s| s.as_str())
        .unwrap_or("");

    let csignal = transcript_sibling(transcript_path, "csignal");

    let signal = match hook_event {
        "PermissionRequest" => {
            let tool_name = input
                .get("tool_name")
                .and_then(|s| s.as_str())
                .unwrap_or("");
            serde_json::json!({
                "type": "permission_request",
                "tool_name": tool_name
            })
        }
        "Notification" => {
            let notification_type = input
                .get("notification_type")
                .and_then(|s| s.as_str())
                .unwrap_or("");
            match notification_type {
                "permission_prompt" => serde_json::json!({"type": "permission_request"}),
                "elicitation_dialog" => serde_json::json!({"type": "elicitation_dialog"}),
                "idle_prompt" => serde_json::json!({"type": "idle_prompt"}),
                _ => return Ok(()), // Unknown notification type — ignore
            }
        }
        "PreCompact" => serde_json::json!({"type": "pre_compact"}),
        _ => return Ok(()), // Unknown signal hook event — ignore
    };

    let mut data = serde_json::to_string(&signal).map_err(|e| format!("serialize: {}", e))?;
    data.push('\n');
    write_atomic(&csignal, data.as_bytes()).map_err(|e| format!("write csignal: {}", e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Daemon mode
// ---------------------------------------------------------------------------

fn daemon_mode(args: &[String]) -> Result<(), String> {
    let transcript_path = get_arg(args, "--transcript-path")?;
    let pid: u32 = get_arg(args, "--pid")?
        .parse()
        .map_err(|_| "invalid --pid")?;
    let ppid: u32 = get_arg(args, "--ppid")?
        .parse()
        .map_err(|_| "invalid --ppid")?;

    let ctx = SessionContext {
        session_id: get_arg(args, "--session-id")?,
        pid,
        ppid,
        cwd: get_arg(args, "--cwd")?,
        cstatus_path: transcript_sibling(&transcript_path, "cstatus"),
        csignal_path: transcript_sibling(&transcript_path, "csignal"),
        cpid_path: transcript_sibling(&transcript_path, "cpid"),
        clog_path: transcript_sibling(&transcript_path, "clog"),
    };

    // Read existing session_name if present
    let session_name = read_session_name(&ctx.cstatus_path);

    let mut state = DaemonState::new();
    state.session_name = session_name;

    // Write daemon PID file so SessionStart can detect us on resume
    let _ = write_atomic(&ctx.cpid_path, process::id().to_string().as_bytes());

    // Open transcript and process existing content
    let file = match fs::File::open(&transcript_path) {
        Ok(f) => f,
        Err(_) => {
            // Transcript doesn't exist yet — wait for it as long as the parent is alive
            loop {
                thread::sleep(POLL_INTERVAL);
                if !pid_is_alive(pid) {
                    cleanup_and_exit(&ctx);
                    return Ok(());
                }
                if Path::new(&transcript_path).exists() {
                    break;
                }
            }
            fs::File::open(&transcript_path).map_err(|e| format!("open transcript: {}", e))?
        }
    };

    let mut reader = BufReader::new(file);
    let mut line_buf = String::new();
    let mut partial_buf = String::new();

    // Process existing lines to establish baseline
    loop {
        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = line_buf.trim_end();
                if !trimmed.is_empty() {
                    state.process_line(trimmed);
                }
            }
            Err(_) => break,
        }
    }

    // Write current state and notify
    write_status(&ctx, &state);
    post_darwin_notification();

    // Enter poll loop
    loop {
        thread::sleep(POLL_INTERVAL);

        // Check liveness
        if !pid_is_alive(pid) {
            cleanup_and_exit(&ctx);
            return Ok(());
        }

        let mut changed = false;

        // Read new JSONL lines
        loop {
            line_buf.clear();
            match reader.read_line(&mut line_buf) {
                Ok(0) => break,
                Ok(_) => {
                    if line_buf.ends_with('\n') {
                        // Complete line
                        let full_line = if partial_buf.is_empty() {
                            line_buf.trim_end().to_string()
                        } else {
                            let mut full = std::mem::take(&mut partial_buf);
                            full.push_str(line_buf.trim_end());
                            full
                        };
                        if !full_line.is_empty() && state.process_line(&full_line) {
                            changed = true;
                        }
                    } else {
                        // Partial line — buffer it
                        partial_buf.push_str(&line_buf);
                    }
                }
                Err(_) => break,
            }
        }

        // Check for signal file
        if let Ok(signal_data) = fs::read_to_string(&ctx.csignal_path) {
            let _ = fs::remove_file(&ctx.csignal_path);
            if let Ok(signal) = serde_json::from_str::<Value>(&signal_data)
                && state.process_signal(&signal)
            {
                changed = true;
            }
        }

        if changed {
            write_status(&ctx, &state);
            post_darwin_notification();
        }
    }
}

struct SessionContext {
    session_id: String,
    pid: u32,
    ppid: u32,
    cwd: String,
    cstatus_path: PathBuf,
    csignal_path: PathBuf,
    cpid_path: PathBuf,
    clog_path: PathBuf,
}

fn write_status(ctx: &SessionContext, state: &DaemonState) {
    let status = StatusInfo {
        session_id: ctx.session_id.clone(),
        pid: ctx.pid,
        ppid: ctx.ppid,
        state: state.state.clone(),
        activity: state.activity.clone(),
        cwd: ctx.cwd.clone(),
        event: state.event.clone(),
        session_name: state.session_name.clone(),
    };
    let _ = write_atomic(&ctx.cstatus_path, status.to_json().as_bytes());
}

fn read_session_name(cstatus_path: &Path) -> Option<String> {
    let contents = fs::read_to_string(cstatus_path).ok()?;
    let v: Value = serde_json::from_str(&contents).ok()?;
    v.get("session_name")
        .and_then(|s| s.as_str())
        .map(String::from)
}

fn cleanup_and_exit(ctx: &SessionContext) {
    let _ = fs::remove_file(&ctx.cstatus_path);
    let _ = fs::remove_file(&ctx.csignal_path);
    let _ = fs::remove_file(&ctx.cpid_path);
    let _ = fs::remove_file(&ctx.clog_path);
    post_darwin_notification();
}

fn get_arg(args: &[String], name: &str) -> Result<String, String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .ok_or_else(|| format!("missing {}", name))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_help() {
    eprintln!("session-status {VERSION}");
    eprintln!("Claude Status plugin hook daemon for the Claude Status macOS menu bar app");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  session-status              Hook mode (reads JSON from stdin, invoked by Claude Code)");
    eprintln!("  session-status --signal     Signal mode (writes .csignal for running daemon)");
    eprintln!("  session-status --daemon     Daemon mode (tails JSONL transcript, maintains state)");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -h, --help       Print this help message");
    eprintln!("  -V, --version    Print version");
    eprintln!();
    eprintln!("This binary is not intended to be run manually. It is invoked by Claude Code");
    eprintln!("hooks registered in plugins/claude-status/hooks/hooks.json.");
}

fn read_stdin_json() -> Result<Value, String> {
    let reader = BufReader::new(io::stdin().lock());
    // Use streaming deserializer — returns as soon as a complete JSON value is
    // parsed, without waiting for EOF on the pipe.
    let mut stream = serde_json::Deserializer::from_reader(reader).into_iter::<Value>();
    match stream.next() {
        Some(Ok(value)) => Ok(value),
        Some(Err(e)) => Err(format!("parse stdin: {}", e)),
        None => Err("parse stdin: empty input".to_string()),
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();

    // Help / version
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        eprintln!("session-status {VERSION}");
        return Ok(());
    }

    // Daemon mode
    if args.iter().any(|a| a == "--daemon") {
        return daemon_mode(&args);
    }

    // Signal mode
    if args.iter().any(|a| a == "--signal") {
        let input = read_stdin_json()?;
        return signal_mode(&input);
    }

    // Hook mode — requires piped stdin
    if io::stdin().is_terminal() {
        print_help();
        return Err("no input provided (stdin is a terminal)".to_string());
    }

    let input = read_stdin_json()?;
    let hook_event = input
        .get("hook_event_name")
        .and_then(|s| s.as_str())
        .unwrap_or("");

    match hook_event {
        "SessionStart" => hook_session_start(&input),
        "SessionEnd" => hook_session_end(&input),
        _ => Ok(()), // Unknown hook event in hook mode — ignore
    }
}

fn main() {
    if let Err(msg) = run() {
        eprintln!("session-status: {}", msg);
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // -- State machine unit tests --

    fn make_assistant_tool_use(tool_name: &str, tool_id: &str) -> String {
        serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": tool_name, "id": tool_id, "input": {}}
                ],
                "stop_reason": "tool_use",
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string()
    }

    fn make_assistant_agent_spawn(agent_id: &str) -> String {
        serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Agent", "id": agent_id, "input": {"prompt": "do stuff"}}
                ],
                "stop_reason": "tool_use",
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string()
    }

    fn make_tool_result(tool_use_id: &str) -> String {
        serde_json::json!({
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": "ok"}
                ],
                "role": "user"
            }
        })
        .to_string()
    }

    fn make_async_tool_result(tool_use_id: &str) -> String {
        serde_json::json!({
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": [
                        {"type": "text", "text": "Async agent launched successfully."}
                    ]}
                ],
                "role": "user"
            },
            "toolUseResult": {
                "isAsync": true,
                "status": "async_launched",
                "agentId": "agent-abc123"
            }
        })
        .to_string()
    }

    fn make_user_text(text: &str) -> String {
        serde_json::json!({
            "type": "user",
            "promptId": "test-prompt-id",
            "message": {
                "content": [{"type": "text", "text": text}],
                "role": "user"
            }
        })
        .to_string()
    }

    fn make_progress(data_type: &str) -> String {
        serde_json::json!({
            "type": "progress",
            "data": {"type": data_type}
        })
        .to_string()
    }

    fn make_compact_boundary() -> String {
        serde_json::json!({
            "type": "system",
            "subtype": "compact_boundary",
            "content": "Conversation compacted"
        })
        .to_string()
    }

    fn make_assistant_streaming() -> String {
        serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "I'll help"}],
                "stop_reason": null,
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string()
    }

    fn make_end_turn(text: &str) -> String {
        serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string()
    }

    #[test]
    fn user_text_sets_active_thinking() {
        let mut s = DaemonState::new();
        s.process_line(&make_user_text("hello"));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "thinking");
    }

    #[test]
    fn user_prompt_clears_stale_agents() {
        let mut s = DaemonState::new();
        // Simulate an agent that was spawned but never completed
        s.active_agents.insert("toolu_stale".to_string());
        s.state = SessionState::Active;
        s.activity = "subagent".to_string();
        // User types a new prompt — the cancelled agent should be cleared
        s.process_line(&make_user_text("continue"));
        assert!(s.active_agents.is_empty());
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "thinking");
    }

    #[test]
    fn meta_user_message_ignored() {
        let mut s = DaemonState::new();
        // isMeta messages (local command caveats) should not change state
        let meta = serde_json::json!({
            "type": "user",
            "isMeta": true,
            "message": {
                "content": "<local-command-caveat>Caveat: messages below were generated by local commands</local-command-caveat>",
                "role": "user"
            }
        })
        .to_string();
        s.process_line(&meta);
        assert_eq!(s.state, SessionState::Idle);
    }

    #[test]
    fn local_command_user_message_no_prompt_id_ignored() {
        let mut s = DaemonState::new();
        // Local command output without promptId should not trigger "thinking"
        let cmd_output = serde_json::json!({
            "type": "user",
            "message": {
                "content": [{"type": "text", "text": "<local-command-stdout>(no content)</local-command-stdout>"}],
                "role": "user"
            }
        })
        .to_string();
        s.process_line(&cmd_output);
        assert_eq!(s.state, SessionState::Idle);
        assert_eq!(s.activity, "");
    }

    #[test]
    fn assistant_streaming_sets_active() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_streaming());
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "");
    }

    #[test]
    fn assistant_tool_use_sets_active_with_tool_name() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_tool_use("Bash", "toolu_123"));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "Bash");
    }

    #[test]
    fn end_turn_without_question_sets_idle() {
        let mut s = DaemonState::new();
        s.process_line(&make_end_turn("Done."));
        assert_eq!(s.state, SessionState::Idle);
        assert_eq!(s.activity, "");
    }

    #[test]
    fn end_turn_with_question_sets_waiting() {
        let mut s = DaemonState::new();
        s.process_line(&make_end_turn("Shall I continue?"));
        assert_eq!(s.state, SessionState::Waiting);
        assert_eq!(s.activity, "question");
    }

    #[test]
    fn end_turn_with_trailing_whitespace_question() {
        let mut s = DaemonState::new();
        s.process_line(&make_end_turn("Shall I continue?   \n  "));
        assert_eq!(s.state, SessionState::Waiting);
        assert_eq!(s.activity, "question");
    }

    #[test]
    fn question_in_last_paragraph_detected() {
        let mut s = DaemonState::new();
        s.process_line(&make_end_turn(
            "I fixed the bug and deployed.\n\nDo you want me to also update the tests?",
        ));
        assert_eq!(s.state, SessionState::Waiting);
        assert_eq!(s.activity, "question");
    }

    #[test]
    fn question_mid_paragraph_last_paragraph_detected() {
        let mut s = DaemonState::new();
        // Question mark mid-sentence in the last paragraph, text continues after
        s.process_line(&make_end_turn(
            "Here's the summary.\n\nIs this what you wanted? Let me know and I can adjust.",
        ));
        assert_eq!(s.state, SessionState::Waiting);
        assert_eq!(s.activity, "question");
    }

    #[test]
    fn question_only_in_early_paragraph_not_detected() {
        let mut s = DaemonState::new();
        // Question is in an earlier paragraph, last paragraph is a statement
        s.process_line(&make_end_turn(
            "Do you want me to continue?\n\nI went ahead and fixed it anyway. Here's what I changed.",
        ));
        assert_eq!(s.state, SessionState::Idle);
    }

    #[test]
    fn end_turn_no_text_blocks_sets_idle() {
        let mut s = DaemonState::new();
        let line = serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [],
                "stop_reason": "end_turn",
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string();
        s.process_line(&line);
        assert_eq!(s.state, SessionState::Idle);
    }

    #[test]
    fn end_turn_with_empty_text_sets_idle() {
        let mut s = DaemonState::new();
        let line = serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": ""}],
                "stop_reason": "end_turn",
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string();
        s.process_line(&line);
        assert_eq!(s.state, SessionState::Idle);
    }

    // -- Agent tracking tests --

    #[test]
    fn agent_spawn_tracked() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_agent_spawn("toolu_agent1"));
        assert!(s.active_agents.contains("toolu_agent1"));
        assert_eq!(s.state, SessionState::Active);
    }

    #[test]
    fn agent_complete_removes_from_set() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_agent_spawn("toolu_agent1"));
        assert_eq!(s.active_agents.len(), 1);
        s.process_line(&make_tool_result("toolu_agent1"));
        assert!(s.active_agents.is_empty());
    }

    #[test]
    fn end_turn_with_active_agents_stays_active() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_agent_spawn("toolu_agent1"));
        // Now claude finishes its turn but agent still running
        s.process_line(&make_end_turn("I've spawned an agent."));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "subagent");
    }

    #[test]
    fn end_turn_after_all_agents_complete_goes_idle() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_agent_spawn("toolu_agent1"));
        s.process_line(&make_tool_result("toolu_agent1"));
        s.process_line(&make_end_turn("All done."));
        assert_eq!(s.state, SessionState::Idle);
    }

    #[test]
    fn concurrent_agents_tracked() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_agent_spawn("toolu_a1"));
        // second agent in a separate assistant message
        s.process_line(&make_assistant_agent_spawn("toolu_a2"));
        assert_eq!(s.active_agents.len(), 2);

        s.process_line(&make_tool_result("toolu_a1"));
        assert_eq!(s.active_agents.len(), 1);

        // end_turn while one agent still active
        s.process_line(&make_end_turn("Done from main."));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "subagent");

        // second agent completes
        s.process_line(&make_tool_result("toolu_a2"));
        assert!(s.active_agents.is_empty());
    }

    #[test]
    fn async_agent_not_removed_on_launch_result() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_agent_spawn("toolu_agent1"));
        assert_eq!(s.active_agents.len(), 1);
        // Async tool_result should NOT remove the agent
        s.process_line(&make_async_tool_result("toolu_agent1"));
        assert_eq!(s.active_agents.len(), 1);
        // end_turn with async agent still active should stay active
        s.process_line(&make_end_turn("Agent is running in the background."));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "subagent");
    }

    #[test]
    fn async_agent_removed_on_sync_completion() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_agent_spawn("toolu_agent1"));
        // Async launch — agent still tracked
        s.process_line(&make_async_tool_result("toolu_agent1"));
        assert_eq!(s.active_agents.len(), 1);
        // Later, sync tool_result when agent completes
        s.process_line(&make_tool_result("toolu_agent1"));
        assert!(s.active_agents.is_empty());
    }

    // -- Compaction detection tests --
    // compact_boundary fires during compaction. Replayed user/progress context
    // is suppressed. The next assistant response clears the compacting flag.

    fn make_compact_assistant_tool_use(tool_name: &str, tool_id: &str) -> String {
        serde_json::json!({
            "type": "assistant",
            "agentId": "acompact-abc123",
            "message": {
                "content": [
                    {"type": "tool_use", "name": tool_name, "id": tool_id, "input": {}}
                ],
                "stop_reason": "tool_use",
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string()
    }

    #[test]
    fn compact_boundary_sets_compacting() {
        let mut s = DaemonState::new();
        s.process_line(&make_compact_boundary());
        assert_eq!(s.state, SessionState::Compacting);
        assert_eq!(s.activity, "compacting");
        assert!(s.compacting);
    }

    #[test]
    fn compact_boundary_suppresses_user_text() {
        // After compact_boundary, replayed user messages should not change state
        let mut s = DaemonState::new();
        s.process_line(&make_compact_boundary());
        assert_eq!(s.state, SessionState::Compacting);
        s.process_line(&make_user_text("This session is being continued..."));
        assert_eq!(s.state, SessionState::Compacting);
        s.process_line(&make_user_text("/compact"));
        assert_eq!(s.state, SessionState::Compacting);
    }

    #[test]
    fn compact_boundary_suppresses_progress() {
        let mut s = DaemonState::new();
        s.process_line(&make_compact_boundary());
        s.process_line(&make_progress("hook_progress"));
        assert_eq!(s.state, SessionState::Compacting);
    }

    #[test]
    fn compacting_clears_on_assistant_response() {
        // The next assistant response after compaction means context replay is done
        let mut s = DaemonState::new();
        s.process_line(&make_compact_boundary());
        s.process_line(&make_user_text("replayed context"));
        assert_eq!(s.state, SessionState::Compacting);
        s.process_line(&make_assistant_streaming());
        assert_eq!(s.state, SessionState::Active);
        assert!(!s.compacting);
    }

    #[test]
    fn compacting_clears_on_end_turn() {
        let mut s = DaemonState::new();
        s.process_line(&make_compact_boundary());
        s.process_line(&make_user_text("replayed context"));
        s.process_line(&make_end_turn("Here's what we were working on."));
        assert_eq!(s.state, SessionState::Idle);
        assert!(!s.compacting);
    }

    #[test]
    fn user_prompt_after_compaction_works_normally() {
        let mut s = DaemonState::new();
        s.process_line(&make_compact_boundary());
        s.process_line(&make_user_text("replayed context"));
        s.process_line(&make_end_turn("Ready."));
        assert_eq!(s.state, SessionState::Idle);
        // Now a real user prompt should work
        s.process_line(&make_user_text("next prompt"));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "thinking");
    }

    #[test]
    fn acompact_agent_id_sets_compacting() {
        // Long compactions spawn a compact agent with acompact- prefix
        let mut s = DaemonState::new();
        s.process_line(&make_compact_assistant_tool_use("Read", "toolu_1"));
        assert_eq!(s.state, SessionState::Compacting);
        assert_eq!(s.activity, "compacting (Read)");
        assert!(s.compacting);
    }

    #[test]
    fn non_compact_agent_id_stays_active() {
        let mut s = DaemonState::new();
        let line = serde_json::json!({
            "type": "assistant",
            "agentId": "a1234567",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {}}
                ],
                "stop_reason": "tool_use",
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string();
        s.process_line(&line);
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "Bash");
    }

    // -- Signal processing tests --

    #[test]
    fn signal_permission_request() {
        let mut s = DaemonState::new();
        let signal = serde_json::json!({"type": "permission_request", "tool_name": "Bash"});
        assert!(s.process_signal(&signal));
        assert_eq!(s.state, SessionState::Waiting);
        assert_eq!(s.activity, "Bash");
    }

    #[test]
    fn signal_elicitation() {
        let mut s = DaemonState::new();
        let signal = serde_json::json!({"type": "elicitation_dialog"});
        assert!(s.process_signal(&signal));
        assert_eq!(s.state, SessionState::Waiting);
        assert_eq!(s.activity, "");
    }

    #[test]
    fn signal_idle_prompt_from_active_with_no_agents() {
        let mut s = DaemonState::new();
        s.state = SessionState::Active;
        let signal = serde_json::json!({"type": "idle_prompt"});
        assert!(s.process_signal(&signal));
        assert_eq!(s.state, SessionState::Idle);
    }

    #[test]
    fn signal_pre_compact_sets_compacting() {
        let mut s = DaemonState::new();
        s.state = SessionState::Active;
        let signal = serde_json::json!({"type": "pre_compact"});
        assert!(s.process_signal(&signal));
        assert_eq!(s.state, SessionState::Compacting);
        assert_eq!(s.activity, "compacting");
        assert!(s.compacting);
    }

    #[test]
    fn realistic_pre_compact_to_idle() {
        // Full sequence: PreCompact signal → compaction → compact_boundary →
        // replayed context → idle_prompt signal
        let mut s = DaemonState::new();
        s.state = SessionState::Active;

        // PreCompact hook fires
        let signal = serde_json::json!({"type": "pre_compact"});
        s.process_signal(&signal);
        assert_eq!(s.state, SessionState::Compacting);

        // compact_boundary fires (compaction done, stays compacting to suppress replay)
        s.process_line(&make_compact_boundary());
        assert_eq!(s.state, SessionState::Compacting);

        // Replayed context suppressed
        s.process_line(&make_user_text("This session is being continued..."));
        assert_eq!(s.state, SessionState::Compacting);
        s.process_line(&make_progress("hook_progress"));
        assert_eq!(s.state, SessionState::Compacting);

        // idle_prompt signal clears to idle
        let idle = serde_json::json!({"type": "idle_prompt"});
        s.process_signal(&idle);
        assert_eq!(s.state, SessionState::Idle);
        assert!(!s.compacting);
    }

    #[test]
    fn signal_idle_prompt_clears_compacting() {
        let mut s = DaemonState::new();
        s.compacting = true;
        s.state = SessionState::Compacting;
        let signal = serde_json::json!({"type": "idle_prompt"});
        assert!(s.process_signal(&signal));
        assert_eq!(s.state, SessionState::Idle);
        assert!(!s.compacting);
    }

    #[test]
    fn signal_idle_prompt_ignored_when_agents_active() {
        let mut s = DaemonState::new();
        s.state = SessionState::Active;
        s.active_agents.insert("toolu_1".to_string());
        let signal = serde_json::json!({"type": "idle_prompt"});
        assert!(!s.process_signal(&signal));
        assert_eq!(s.state, SessionState::Active);
    }

    // -- turn_duration tests --

    fn make_turn_duration() -> String {
        serde_json::json!({
            "type": "system",
            "subtype": "turn_duration",
            "isMeta": false
        })
        .to_string()
    }

    #[test]
    fn turn_duration_transitions_active_to_idle() {
        let mut s = DaemonState::new();
        // Simulate streaming response (stop_reason: null) leaving state Active
        s.process_line(&make_assistant_streaming());
        assert_eq!(s.state, SessionState::Active);
        // turn_duration fires — should transition to idle
        s.process_line(&make_turn_duration());
        assert_eq!(s.state, SessionState::Idle);
        assert_eq!(s.activity, "");
    }

    #[test]
    fn turn_duration_clears_stale_agents() {
        let mut s = DaemonState::new();
        s.process_line(&make_assistant_agent_spawn("toolu_1"));
        s.process_line(&make_assistant_streaming());
        assert_eq!(s.state, SessionState::Active);
        assert!(!s.active_agents.is_empty());
        // turn_duration should clear stale agents and transition to idle
        s.process_line(&make_turn_duration());
        assert_eq!(s.state, SessionState::Idle);
        assert!(s.active_agents.is_empty());
    }

    #[test]
    fn turn_duration_no_change_when_idle() {
        let mut s = DaemonState::new();
        assert_eq!(s.state, SessionState::Idle);
        let changed = s.process_line(&make_turn_duration());
        assert!(!changed);
        assert_eq!(s.state, SessionState::Idle);
    }

    #[test]
    fn max_tokens_transitions_to_idle() {
        let mut s = DaemonState::new();
        s.state = SessionState::Active;
        let line = serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "truncated response..."}],
                "stop_reason": "max_tokens",
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string();
        s.process_line(&line);
        assert_eq!(s.state, SessionState::Idle);
    }

    // -- Partial line handling tests --

    #[test]
    fn partial_line_not_processed() {
        let mut s = DaemonState::new();
        // Incomplete JSON — should return false and not crash
        let changed = s.process_line("{\"type\": \"assistant\", \"message\":");
        assert!(!changed);
    }

    #[test]
    fn malformed_json_skipped() {
        let mut s = DaemonState::new();
        assert!(!s.process_line("not json at all"));
        assert_eq!(s.state, SessionState::Idle);
    }

    // -- Path derivation tests --

    #[test]
    fn cstatus_path_derived_correctly() {
        let p = transcript_sibling(
            "/Users/test/.claude/projects/-test-project/abc123.jsonl",
            "cstatus",
        );
        assert_eq!(
            p,
            PathBuf::from("/Users/test/.claude/projects/-test-project/abc123.cstatus")
        );
    }

    #[test]
    fn csignal_path_derived_correctly() {
        let p = transcript_sibling(
            "/Users/test/.claude/projects/-test-project/abc123.jsonl",
            "csignal",
        );
        assert_eq!(
            p,
            PathBuf::from("/Users/test/.claude/projects/-test-project/abc123.csignal")
        );
    }

    // -- Atomic write tests --

    #[test]
    fn write_atomic_creates_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.cstatus");
        write_atomic(&path, b"hello\n").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello\n");
    }

    #[test]
    fn write_atomic_overwrites_existing() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.cstatus");
        fs::write(&path, "old").unwrap();
        write_atomic(&path, b"new\n").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "new\n");
    }

    // -- StatusInfo JSON tests --

    #[test]
    fn status_json_format() {
        let s = StatusInfo {
            session_id: "abc".to_string(),
            pid: 123,
            ppid: 456,
            state: SessionState::Active,
            activity: "thinking".to_string(),
            cwd: "/tmp".to_string(),
            event: "user".to_string(),
            session_name: None,
        };
        let json = s.to_json();
        let v: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["session_id"], "abc");
        assert_eq!(v["pid"], 123);
        assert_eq!(v["ppid"], 456);
        assert_eq!(v["state"], "active");
        assert_eq!(v["activity"], "thinking");
        assert_eq!(v["cwd"], "/tmp");
        assert_eq!(v["event"], "user");
        assert!(v.get("session_name").is_none());
    }

    #[test]
    fn status_json_with_session_name() {
        let s = StatusInfo {
            session_id: "abc".to_string(),
            pid: 123,
            ppid: 456,
            state: SessionState::Idle,
            activity: String::new(),
            cwd: "/tmp".to_string(),
            event: "SessionStart".to_string(),
            session_name: Some("My Session".to_string()),
        };
        let json = s.to_json();
        let v: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["session_name"], "My Session");
    }

    // -- PID liveness (basic) --

    #[test]
    fn current_process_is_alive() {
        assert!(pid_is_alive(process::id()));
    }

    #[test]
    fn nonexistent_pid_is_not_alive() {
        assert!(!pid_is_alive(4_000_000));
    }

    // -- get_ppid_of --

    #[test]
    fn get_ppid_of_current_process() {
        let ppid = get_ppid_of(process::id());
        assert!(ppid.is_some());
        assert!(ppid.unwrap() > 0);
    }

    #[test]
    fn get_ppid_of_nonexistent() {
        assert!(get_ppid_of(4_000_000).is_none());
    }

    // -- Integration-style: process a realistic sequence --

    #[test]
    fn realistic_sequence() {
        let mut s = DaemonState::new();

        // User submits prompt
        s.process_line(&make_user_text("Fix the bug"));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "thinking");

        // Assistant starts streaming
        s.process_line(&make_assistant_streaming());
        assert_eq!(s.state, SessionState::Active);

        // Assistant calls a tool
        s.process_line(&make_assistant_tool_use("Read", "toolu_read1"));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "Read");

        // Bash progress while tool runs
        s.process_line(&make_progress("bash_progress"));
        assert_eq!(s.activity, "bash");

        // Tool result comes back
        s.process_line(&make_tool_result("toolu_read1"));
        // tool_result doesn't change state

        // Assistant responds with text end_turn
        s.process_line(&make_end_turn("I've fixed the bug."));
        assert_eq!(s.state, SessionState::Idle);
        assert_eq!(s.activity, "");
    }

    #[test]
    fn realistic_sequence_with_question() {
        let mut s = DaemonState::new();

        s.process_line(&make_user_text("Help me"));
        s.process_line(&make_end_turn("Would you like me to proceed?"));
        assert_eq!(s.state, SessionState::Waiting);
        assert_eq!(s.activity, "question");
    }

    #[test]
    fn realistic_manual_compact_sequence() {
        // Manual /compact: compact_boundary, replayed context, then idle
        let mut s = DaemonState::new();

        s.process_line(&make_user_text("do something"));
        assert_eq!(s.state, SessionState::Active);

        // compact_boundary fires
        s.process_line(&make_compact_boundary());
        assert_eq!(s.state, SessionState::Compacting);

        // Replayed context — suppressed
        s.process_line(&make_user_text("This session is being continued..."));
        assert_eq!(s.state, SessionState::Compacting);
        s.process_line(&make_user_text("/compact"));
        assert_eq!(s.state, SessionState::Compacting);
        s.process_line(&make_progress("hook_progress"));
        assert_eq!(s.state, SessionState::Compacting);

        // Assistant responds post-compaction
        s.process_line(&make_end_turn("Context compacted. Ready to continue."));
        assert_eq!(s.state, SessionState::Idle);
        assert!(!s.compacting);

        // Next user prompt works normally
        s.process_line(&make_user_text("next prompt"));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "thinking");
    }

    #[test]
    fn realistic_long_compact_with_agent() {
        // Long compaction that spawns a compact agent (acompact- agentId)
        let mut s = DaemonState::new();

        // Compact agent tool use detected by agentId
        s.process_line(&make_compact_assistant_tool_use("Read", "toolu_1"));
        assert_eq!(s.state, SessionState::Compacting);
        assert!(s.compacting);

        // compact_boundary fires (stays compacting)
        s.process_line(&make_compact_boundary());
        assert_eq!(s.state, SessionState::Compacting);

        // Replayed context — suppressed
        s.process_line(&make_user_text("replayed summary"));
        assert_eq!(s.state, SessionState::Compacting);

        // Assistant responds
        s.process_line(&make_end_turn("Done."));
        assert_eq!(s.state, SessionState::Idle);
        assert!(!s.compacting);
    }

    // -- Signal file integration --

    #[test]
    fn signal_file_round_trip() {
        let tmp = TempDir::new().unwrap();
        let csignal = tmp.path().join("test.csignal");

        let signal = serde_json::json!({"type": "permission_request", "tool_name": "Bash"});
        let data = serde_json::to_string(&signal).unwrap() + "\n";
        write_atomic(&csignal, data.as_bytes()).unwrap();

        let read_back = fs::read_to_string(&csignal).unwrap();
        let parsed: Value = serde_json::from_str(&read_back).unwrap();
        assert_eq!(parsed["type"], "permission_request");
        assert_eq!(parsed["tool_name"], "Bash");

        fs::remove_file(&csignal).unwrap();
        assert!(!csignal.exists());
    }

    // -- Mixed content assistant message (text + tool_use) --

    #[test]
    fn assistant_text_plus_tool_use() {
        let mut s = DaemonState::new();
        let line = serde_json::json!({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "name": "Bash", "id": "toolu_1", "input": {"command": "ls"}}
                ],
                "stop_reason": "tool_use",
                "type": "message",
                "role": "assistant"
            }
        })
        .to_string();
        s.process_line(&line);
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "Bash");
    }

    // -- Agent progress sets active subagent --

    #[test]
    fn agent_progress_sets_subagent() {
        let mut s = DaemonState::new();
        s.process_line(&make_progress("agent_progress"));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "subagent");
    }

    // -- MCP progress --

    #[test]
    fn mcp_progress_sets_mcp() {
        let mut s = DaemonState::new();
        s.process_line(&make_progress("mcp_progress"));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "mcp");
    }

    // -- No-op types don't change state --

    #[test]
    fn file_history_snapshot_noop() {
        let mut s = DaemonState::new();
        s.state = SessionState::Active;
        s.activity = "thinking".to_string();
        let line = serde_json::json!({"type": "file-history-snapshot"}).to_string();
        let changed = s.process_line(&line);
        assert!(!changed);
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "thinking");
    }

    #[test]
    fn hook_progress_noop() {
        let mut s = DaemonState::new();
        s.state = SessionState::Active;
        s.activity = "Bash".to_string();
        s.process_line(&make_progress("hook_progress"));
        assert_eq!(s.state, SessionState::Active);
        assert_eq!(s.activity, "Bash");
    }

    #[test]
    fn stop_hook_summary_noop() {
        let mut s = DaemonState::new();
        s.state = SessionState::Idle;
        let line = serde_json::json!({
            "type": "system",
            "subtype": "stop_hook_summary"
        })
        .to_string();
        let changed = s.process_line(&line);
        assert!(!changed);
    }

    #[test]
    fn turn_duration_noop() {
        let mut s = DaemonState::new();
        s.state = SessionState::Idle;
        let line = serde_json::json!({
            "type": "system",
            "subtype": "turn_duration"
        })
        .to_string();
        let changed = s.process_line(&line);
        assert!(!changed);
    }
}
