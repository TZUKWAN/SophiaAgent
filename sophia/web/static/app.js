/* SophiaAgent -- Frontend Application Logic */

const App = {
    state: {
        sessionId: null,
        sessions: [],
        loading: false,
        ws: null,
        currentAssistantEl: null,
        currentTextEl: null,
        currentActivityEl: null,
        currentStreamText: '',
        currentWorkspace: localStorage.getItem('sophia-workspace') || '',
    },

    init() {
        this.bindEvents();
        this.connectWebSocket();

        // Workspace gate: must select workspace before chatting
        if (this.state.currentWorkspace) {
            const shortName = this.state.currentWorkspace.split(/[\\/]/).pop() || this.state.currentWorkspace;
            const subtitle = document.getElementById('logoSubtitle');
            if (subtitle) subtitle.textContent = shortName;
            this.selectWorkspace(this.state.currentWorkspace, false);
        } else {
            this.showWorkspaceModal();
        }
    },

    bindEvents() {
        const $ = id => document.getElementById(id);

        // Close WebSocket cleanly on page unload so the server can shut down
        window.addEventListener('beforeunload', () => {
            if (this.state.ws) {
                this.state.ws.onclose = null; // prevent auto-reconnect
                this.state.ws.close();
            }
        });

        $('newChatBtn').onclick = () => this.newChat();
        $('settingsBtn').onclick = () => this.openSettings();
        $('settingsClose').onclick = () => this.closeSettings();
        $('settingsOverlay').onclick = () => this.closeSettings();
        $('saveSettingsBtn').onclick = () => this.saveSettings();
        $('sidebarToggle').onclick = () => $('sidebar').classList.toggle('open');

        const input = $('messageInput');
        input.oninput = () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 180) + 'px';
            $('sendBtn').disabled = !input.value.trim();
        };
        input.onkeydown = (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (input.value.trim()) this.sendMessage();
            }
        };
        $('sendBtn').onclick = () => this.sendMessage();

        $('addWorkspaceBtn').onclick = () => {
            const path = prompt('Enter workspace path:');
            if (path && path.trim()) {
                this.addWorkspace(path.trim());
            }
        };

        $('switchWorkspaceBtn').onclick = () => this.showWorkspaceModal();
        $('modalAddWorkspaceBtn').onclick = () => {
            const path = prompt('Enter workspace path:');
            if (path && path.trim()) {
                this.addWorkspace(path.trim()).then(() => this.loadWorkspacesForModal());
            }
        };
    },

    // ── WebSocket ──────────────────────────

    connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${location.host}/api/chat/stream`);

        ws.onopen = () => console.log('WebSocket connected');
        ws.onclose = () => {
            console.log('WebSocket closed, reconnecting in 2s');
            setTimeout(() => this.connectWebSocket(), 2000);
        };
        ws.onerror = () => ws.close();
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleStreamEvent(data);
            } catch(e) { console.error('WS parse error', e); }
        };
        this.state.ws = ws;
    },

    handleStreamEvent(data) {
        switch (data.type) {
            case 'token':
                this.appendToCurrentMessage(data.content);
                break;
            case 'tool_call':
                this.createToolCard(data.name, 'running');
                break;
            case 'tool_result':
                this.updateToolCard(data.name, data.result, 'success');
                break;
            case 'workspace_scan_start':
                this.createToolCard('workspace', 'running');
                this.updateToolCard('workspace', `Scanning ${data.total_files || 0} workspace files...`, 'running');
                break;
            case 'workspace_file_start':
                this.updateToolCard('workspace', `[${data.index}/${data.total}] Reading ${data.path}`, 'running');
                break;
            case 'workspace_file_done':
                this.updateToolCard('workspace', `[${data.index}/${data.total}] ${data.status}: ${data.path} (${data.chars || 0} chars)`, data.status === 'read' ? 'running' : 'error');
                break;
            case 'workspace_scan_done':
                this.updateToolCard('workspace', `Read ${data.read_files || 0}/${data.total_files || 0} workspace files. Skipped ${data.skipped_files || 0}.`, 'success');
                break;
            case 'swarm_analyze':
                this.createToolCard('swarm', 'running');
                this.updateToolCard('swarm', data.reason || 'Analyzing task complexity...', 'running');
                break;
            case 'swarm_plan':
                this.updateToolCard('swarm', `Plan: ${data.workflow || 'mixed'} \u00b7 ${data.stages ? data.stages.length : 0} stages`, 'running');
                break;
            case 'swarm_stage_start':
                this.createToolCard(`swarm:${data.stage_id}`, 'running');
                break;
            case 'swarm_agent_start':
                this.createToolCard(`agent:${data.agent_id}`, 'running');
                this.updateToolCard(`agent:${data.agent_id}`, `${data.role_id || data.agent_id} is working...`, 'running');
                break;
            case 'swarm_agent_complete':
                this.updateToolCard(`agent:${data.agent_id}`, `${data.role_id || data.agent_id}: ${data.status}`, data.status === 'completed' ? 'success' : 'error');
                break;
            case 'swarm_agent_error':
                this.updateToolCard(`agent:${data.agent_id}`, data.error || 'Agent failed', 'error');
                break;
            case 'swarm_stage_end':
                this.updateToolCard(`swarm:${data.stage_id}`, 'Stage complete', 'success');
                break;
            case 'swarm_synthesize':
                this.updateToolCard('swarm', `Synthesizing ${data.agent_count || 0} expert outputs...`, 'running');
                break;
            case 'swarm_done':
                this.updateToolCard('swarm', 'Swarm complete', 'success');
                break;
            case 'done':
                this.finalizeMessage(data.content || '');
                if (data.session_id) this.state.sessionId = data.session_id;
                this.state.loading = false;
                this.loadSessions();
                break;
            case 'error':
                this.finalizeMessage('');
                this.showError(data.error);
                this.state.loading = false;
                break;
        }
    },

    // ── Messaging ──────────────────────────

    sendMessage() {
        const input = document.getElementById('messageInput');
        const text = input.value.trim();
        if (!text || this.state.loading) return;

        const empty = document.getElementById('emptyState');
        if (empty) empty.style.display = 'none';

        this.appendMessage('user', text);
        input.value = '';
        input.style.height = 'auto';
        document.getElementById('sendBtn').disabled = true;

        this.state.loading = true;

        const ws = this.state.ws;
        if (ws && ws.readyState === WebSocket.OPEN) {
            this.startAssistantMessage();
            ws.send(JSON.stringify({
                message: text,
                session_id: this.state.sessionId || '',
            }));
        } else {
            this.sendSync(text);
        }
    },

    async sendSync(text) {
        this.startAssistantMessage();
        try {
            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    message: text,
                    session_id: this.state.sessionId || '',
                }),
            });
            const data = await resp.json();
            if (data.session_id) this.state.sessionId = data.session_id;
            this.finalizeMessage(data.response || '');
            this.loadSessions();
        } catch (e) {
            this.finalizeMessage('');
            this.showError(e.message);
        }
        this.state.loading = false;
    },

    appendMessage(role, content) {
        const container = document.getElementById('messages');
        const div = document.createElement('div');
        div.className = `message message-${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'S';

        const body = document.createElement('div');
        body.className = 'message-body';

        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        if (role === 'user') {
            contentEl.textContent = content;
        } else {
            contentEl.innerHTML = this.renderMarkdown(content);
        }

        body.appendChild(contentEl);
        div.appendChild(avatar);
        div.appendChild(body);
        container.appendChild(div);
        this.scrollToBottom();
    },

    startAssistantMessage() {
        const container = document.getElementById('messages');
        const div = document.createElement('div');
        div.className = 'message message-assistant';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'S';

        const body = document.createElement('div');
        body.className = 'message-body';

        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        const activityEl = document.createElement('div');
        activityEl.className = 'activity-stack';
        const textEl = document.createElement('div');
        textEl.className = 'assistant-text';
        contentEl.appendChild(activityEl);
        contentEl.appendChild(textEl);

        body.appendChild(contentEl);
        div.appendChild(avatar);
        div.appendChild(body);
        container.appendChild(div);

        this.state.currentAssistantEl = contentEl;
        this.state.currentTextEl = textEl;
        this.state.currentActivityEl = activityEl;
        this.state.currentStreamText = '';
    },

    appendToCurrentMessage(chunk) {
        this.state.currentStreamText += chunk;
        if (this.state.currentTextEl) {
            this.state.currentTextEl.innerHTML = this.renderMarkdown(this.state.currentStreamText);
            this.scrollToBottom();
        }
    },

    finalizeMessage(text) {
        if (this.state.currentAssistantEl && this.state.currentTextEl) {
            const final = text || this.state.currentStreamText;
            this.state.currentTextEl.innerHTML = this.renderMarkdown(final);
            this.renderMath(this.state.currentTextEl);
            this.highlightCode(this.state.currentTextEl);
            this.state.currentAssistantEl = null;
            this.state.currentTextEl = null;
            this.state.currentActivityEl = null;
            this.state.currentStreamText = '';
            this.scrollToBottom();
        }
    },

    createToolCard(name, status) {
        if (!this.state.currentAssistantEl) return;
        const parent = this.state.currentActivityEl || this.state.currentAssistantEl;
        const existing = parent.querySelector(`.tool-card[data-tool-name="${CSS.escape(name)}"]`);
        if (existing) return;

        const card = document.createElement('div');
        card.className = 'tool-card';
        card.dataset.toolName = name;

        const header = document.createElement('div');
        header.className = 'tool-card-header';
        header.innerHTML = `<span class="dot dot-${status}"></span> ${name}`;
        header.onclick = () => card.classList.toggle('collapsed');

        const body = document.createElement('div');
        body.className = 'tool-card-body';
        body.textContent = 'Running...';

        card.appendChild(header);
        card.appendChild(body);
        parent.appendChild(card);
        this.scrollToBottom();
    },

    updateToolCard(name, result, status) {
        const cards = document.querySelectorAll('.tool-card');
        let found = false;
        for (const card of cards) {
            if (card.dataset.toolName === name) {
                found = true;
                const dot = card.querySelector('.dot');
                if (dot) { dot.className = `dot dot-${status}`; }
                const body = card.querySelector('.tool-card-body');
                if (body) { body.textContent = result || '(no output)'; }
            }
        }
        if (!found && this.state.currentAssistantEl) {
            this.createToolCard(name, status || 'running');
            this.updateToolCard(name, result, status);
        }
        this.scrollToBottom();
    },

    renderMarkdown(text) {
        if (!text) return '';
        try {
            return marked.parse(text, { breaks: true, gfm: true });
        } catch(e) {
            return text.replace(/\n/g, '<br>');
        }
    },

    renderMath(container) {
        try {
            renderMathInElement(container, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                ],
                throwOnError: false,
            });
        } catch(e) {}
    },

    highlightCode(container) {
        container.querySelectorAll('pre code').forEach(block => {
            try { hljs.highlightElement(block); } catch(e) {}
        });
    },

    showError(msg) {
        this.appendMessage('assistant', `**Error:** ${msg}`);
    },

    scrollToBottom() {
        const el = document.getElementById('messages');
        el.scrollTop = el.scrollHeight;
    },

    // ── Sessions ───────────────────────────

    async loadSessions() {
        try {
            const resp = await fetch('/api/sessions');
            const data = await resp.json();
            this.state.sessions = data.sessions || [];
            this.renderSessions();
        } catch(e) { console.error('Load sessions failed', e); }
    },

    renderSessions() {
        const container = document.getElementById('sessionsList');
        container.innerHTML = '';

        const grouped = {};
        this.state.sessions.forEach(s => {
            const ws = s.workspace || 'Default';
            if (!grouped[ws]) grouped[ws] = [];
            grouped[ws].push(s);
        });

        const keys = Object.keys(grouped);
        if (keys.length === 0) {
            container.innerHTML = '<div style="padding:12px 16px;color:var(--text-dim);font-size:12px;">No conversations yet</div>';
            return;
        }

        keys.forEach(workspace => {
            const section = document.createElement('div');
            section.className = 'workspace-group';

            const header = document.createElement('div');
            header.className = 'workspace-header';
            header.innerHTML = `<span class="workspace-arrow">&#9662;</span>
                               <span class="workspace-name" title="${workspace}">${this.truncatePath(workspace)}</span>`;
            header.onclick = () => {
                header.classList.toggle('collapsed');
            };

            const list = document.createElement('div');
            list.className = 'workspace-sessions';

            grouped[workspace].forEach(s => {
                const item = document.createElement('div');
                item.className = 'session-item' + (s.id === this.state.sessionId ? ' active' : '');

                const title = document.createElement('span');
                title.className = 'session-item-title';
                title.textContent = s.title || 'Untitled';
                title.title = s.title || '';
                title.onclick = () => this.selectSession(s.id);

                const del = document.createElement('button');
                del.className = 'session-delete';
                del.textContent = '\u00d7';
                del.onclick = (e) => { e.stopPropagation(); this.deleteSession(s.id); };

                item.appendChild(title);
                item.appendChild(del);
                list.appendChild(item);
            });

            section.appendChild(header);
            section.appendChild(list);
            container.appendChild(section);
        });
    },

    truncatePath(path, maxLen = 28) {
        if (path.length <= maxLen) return path;
        const name = path.split(/[\\/]/).pop() || path;
        if (name.length <= maxLen) return name;
        return '...' + path.slice(-(maxLen - 3));
    },

    async selectSession(sessionId) {
        try {
            const resp = await fetch(`/api/sessions/${sessionId}`);
            const data = await resp.json();

            this.state.sessionId = sessionId;
            const messages = document.getElementById('messages');
            messages.innerHTML = '';

            const empty = document.getElementById('emptyState');
            if (empty) empty.style.display = 'none';

            (data.messages || []).forEach(msg => {
                if (msg.role === 'user' || msg.role === 'assistant') {
                    this.appendMessage(msg.role, msg.content || '');
                }
            });

            this.renderSessions();
        } catch(e) { console.error('Select session failed', e); }
    },

    async deleteSession(sessionId) {
        if (!confirm('Delete this conversation?')) return;
        try {
            await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
            if (this.state.sessionId === sessionId) {
                this.state.sessionId = null;
                this.newChat();
            }
            this.loadSessions();
        } catch(e) { console.error('Delete session failed', e); }
    },

    newChat() {
        this.state.sessionId = null;
        const messages = document.getElementById('messages');
        messages.innerHTML = '';
        const empty = document.getElementById('emptyState');
        if (empty) {
            empty.style.display = '';
            messages.appendChild(empty);
        }
        this.renderSessions();
    },

    // ── Workspace Selection ────────────────

    showWorkspaceModal() {
        document.getElementById('workspaceModalOverlay').classList.add('active');
        this.loadWorkspacesForModal();
    },

    hideWorkspaceModal() {
        document.getElementById('workspaceModalOverlay').classList.remove('active');
    },

    async loadWorkspacesForModal() {
        try {
            const resp = await fetch('/api/workspaces');
            const data = await resp.json();
            this.renderWorkspaceModal(data.workspaces || []);
        } catch(e) { console.error('Load workspaces failed', e); }
    },

    renderWorkspaceModal(workspaces) {
        const grid = document.getElementById('workspaceGrid');
        grid.innerHTML = '';

        if (workspaces.length === 0) {
            grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;color:var(--text-dim);padding:20px;font-size:13px;">No workspaces found. Add one below.</div>';
            return;
        }

        workspaces.forEach(path => {
            const card = document.createElement('div');
            card.className = 'workspace-card' + (path === this.state.currentWorkspace ? ' active' : '');
            card.onclick = () => this.selectWorkspace(path);

            const name = path.split(/[\\/]/).pop() || path;

            card.innerHTML = `
                <div class="card-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
                </div>
                <div class="card-meta">
                    <div class="card-name" title="${name}">${name}</div>
                    <div class="card-path" title="${path}">${path}</div>
                </div>
                <div class="card-check">
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                </div>
            `;
            grid.appendChild(card);
        });
    },

    async selectWorkspace(path, hideModal = true) {
        try {
            const resp = await fetch('/api/workspace/switch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({workspace: path}),
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Switch failed');
            }
            localStorage.setItem('sophia-workspace', path);
            this.state.currentWorkspace = path;
            const shortName = path.split(/[\\/]/).pop() || path;
            const subtitle = document.getElementById('logoSubtitle');
            if (subtitle) subtitle.textContent = shortName;
            if (hideModal) this.hideWorkspaceModal();
            this.newChat();
            this.loadSessions();
            this.showToast(`Workspace: ${shortName}`);
        } catch(e) {
            this.showToast('Error: ' + e.message);
            console.error('Switch workspace failed', e);
        }
    },

    // ── Settings ───────────────────────────

    async openSettings() {
        document.getElementById('settingsOverlay').classList.add('active');
        document.getElementById('settingsPanel').classList.add('active');
        await this.loadSettings();
    },

    closeSettings() {
        document.getElementById('settingsOverlay').classList.remove('active');
        document.getElementById('settingsPanel').classList.remove('active');
    },

    async loadSettings() {
        try {
            const resp = await fetch('/api/settings');
            const data = await resp.json();
            document.getElementById('settingProvider').value = data.provider || 'openai-compat';
            document.getElementById('settingModel').value = data.model || '';
            document.getElementById('settingBaseUrl').value = data.base_url || '';
            document.getElementById('settingApiKey').value = '';
            document.getElementById('settingApiKey').placeholder = data.api_key_masked || 'sk-...';

            const wsResp = await fetch('/api/workspaces');
            const wsData = await wsResp.json();
            const select = document.getElementById('settingWorkspace');
            select.innerHTML = '';
            (wsData.workspaces || []).forEach(w => {
                const opt = document.createElement('option');
                opt.value = w;
                opt.textContent = w;
                if (w === data.workspace) opt.selected = true;
                select.appendChild(opt);
            });
            if (data.workspace) {
                const shortName = data.workspace.split(/[\\/]/).pop() || data.workspace;
                const subtitle = document.getElementById('logoSubtitle');
                if (subtitle) subtitle.textContent = shortName;
            }
        } catch(e) { console.error('Load settings failed', e); }
    },

    async saveSettings() {
        const body = {
            provider: document.getElementById('settingProvider').value,
            model: document.getElementById('settingModel').value,
            base_url: document.getElementById('settingBaseUrl').value,
            workspace: document.getElementById('settingWorkspace').value,
        };
        const apiKey = document.getElementById('settingApiKey').value;
        if (apiKey) body.api_key = apiKey;

        try {
            const resp = await fetch('/api/settings', {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                const data = await resp.json();
                document.getElementById('modelBadge').textContent = data.settings.model;
                const newWs = data.settings.workspace;
                if (newWs && newWs !== this.state.currentWorkspace) {
                    localStorage.setItem('sophia-workspace', newWs);
                    this.state.currentWorkspace = newWs;
                    this.newChat();
                }
                this.showToast('Settings saved');
                this.closeSettings();
                this.loadSessions();
            } else {
                this.showToast('Failed to save settings');
            }
        } catch(e) {
            this.showToast('Error: ' + e.message);
        }
    },

    async addWorkspace(path) {
        try {
            const resp = await fetch('/api/workspaces', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ path }),
            });
            if (resp.ok) {
                this.showToast('Workspace added');
                this.loadSettings();
            }
        } catch(e) { this.showToast('Error: ' + e.message); }
    },

    // ── Toast ──────────────────────────────

    showToast(msg) {
        const el = document.getElementById('toast');
        el.textContent = msg;
        el.classList.add('show');
        setTimeout(() => el.classList.remove('show'), 2500);
    },
};

document.addEventListener('DOMContentLoaded', () => App.init());
