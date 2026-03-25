// ========================================
// VAVE AI - Main Application JavaScript
// ========================================

class ChatbotApp {
    constructor() {
        this.currentSection = 'chat';
        this.init();
    }

    init() {
        this.bindEvents();
        this.initTheme();
        this.handleURLParams();
    }

    // ========================================
    // Event Bindings
    // ========================================

    bindEvents() {
        // Navigation
        document.querySelectorAll('.nav-item[data-section]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchSection(item.dataset.section);
            });
        });

        // Sidebar toggle
        const sidebarToggle = document.getElementById('sidebarToggle');
        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', () => {
                document.querySelector('.sidebar').classList.toggle('collapsed');
            });
        }

        // Theme toggle
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        // Theme select
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            themeSelect.value = localStorage.getItem('theme') || 'light';
            themeSelect.addEventListener('change', (e) => {
                const theme = e.target.value;
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('theme', theme);
                this.updateThemeIcon(theme);
            });
        }

        // User menu toggle
        const userMenuToggle = document.getElementById('userMenuToggle');
        const userDropdown = document.getElementById('userDropdown');
        if (userMenuToggle && userDropdown) {
            userMenuToggle.addEventListener('click', (e) => {
                e.stopPropagation();
                userDropdown.classList.toggle('active');
            });

            // Close dropdown when clicking outside
            document.addEventListener('click', () => {
                userDropdown.classList.remove('active');
            });

            userDropdown.addEventListener('click', (e) => {
                e.stopPropagation();
            });
        }

        // Chat input
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');

        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            // Auto-resize textarea
            chatInput.addEventListener('input', (e) => {
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
            });
        }

        // Image upload
        const imageInput = document.getElementById('imageInput');
        if (imageInput) {
            imageInput.addEventListener('change', (e) => {
                if (e.target.files[0]) {
                    document.getElementById('imageFileName').textContent = e.target.files[0].name;
                    document.getElementById('imagePreviewContainer').classList.remove('hidden');
                }
            });
        }

        const clearImageBtn = document.getElementById('clearImageBtn');
        if (clearImageBtn) {
            clearImageBtn.addEventListener('click', () => {
                imageInput.value = '';
                document.getElementById('imagePreviewContainer').classList.add('hidden');
            });
        }

        // Database file inputs - with dropzone visual feedback
        const databaseExcelFile = document.getElementById('databaseExcelFile');
        if (databaseExcelFile) {
            databaseExcelFile.addEventListener('change', (e) => {
                const file = e.target.files[0];
                const label = document.getElementById('databaseExcelFileName');
                const zone = document.getElementById('excelDropzone');
                if (file) {
                    label.textContent = file.name;
                    zone && zone.classList.add('has-file');
                } else {
                    label.textContent = 'No file selected';
                    zone && zone.classList.remove('has-file');
                }
            });
        }

        const databaseZipFile = document.getElementById('databaseZipFile');
        if (databaseZipFile) {
            databaseZipFile.addEventListener('change', (e) => {
                const file = e.target.files[0];
                const label = document.getElementById('databaseZipFileName');
                const zone = document.getElementById('zipDropzone');
                if (file) {
                    label.textContent = file.name;
                    zone && zone.classList.add('has-file');
                } else {
                    label.textContent = 'No file selected';
                    zone && zone.classList.remove('has-file');
                }
            });
        }

        // Database upload form
        const dbUploadForm = document.getElementById('databaseUploadForm');
        if (dbUploadForm) {
            dbUploadForm.addEventListener('submit', (e) => this.handleDatabaseUpload(e));
        }

        // Change password form
        const passwordForm = document.getElementById('changePasswordForm');
        if (passwordForm) {
            passwordForm.addEventListener('submit', (e) => this.handlePasswordChange(e));
        }
        // Create user form
        const createUserForm = document.getElementById('createUserForm');
        if (createUserForm) {
            createUserForm.addEventListener('submit', (e) => this.handleCreateUser(e));
        }
    }

    // ========================================
    // Theme Management
    // ========================================

    initTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    }

    toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        this.updateThemeIcon(next);

        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            themeSelect.value = next;
        }
    }

    updateThemeIcon(theme) {
        const icon = document.querySelector('#themeToggle i');
        if (icon) {
            icon.className = theme === 'dark' ? 'ph ph-sun' : 'ph ph-moon';
        }
    }

    // ========================================
    // Navigation
    // ========================================

    handleURLParams() {
        const urlParams = new URLSearchParams(window.location.search);
        const section = urlParams.get('section');

        if (section) {
            this.switchSection(section);
            window.history.replaceState({}, document.title, window.location.pathname);
        }
    }

    switchSection(sectionName) {
        // Update content sections
        document.querySelectorAll('.content-section').forEach(el => {
            el.classList.remove('active');
        });

        // Update navigation items
        document.querySelectorAll('.nav-item').forEach(el => {
            el.classList.remove('active');
        });

        const targetSection = document.getElementById(`${sectionName}Section`);
        if (targetSection) {
            targetSection.classList.add('active');
        }

        const navItem = document.querySelector(`[data-section="${sectionName}"]`);
        if (navItem) {
            navItem.classList.add('active');
        }

        this.currentSection = sectionName;

        // Load data for specific sections
        if (sectionName === 'history') {
            this.loadHistory();
        } else if (sectionName === 'logs') {
            this.loadLogs();
        } else if (sectionName === 'database') {
            this.loadDatabaseStatus();
            this.loadDatabaseUploads();
        } else if (sectionName === 'analytics') {
            this.loadAnalytics();
        }
    }

    // ========================================
    // Chat Functionality
    // ========================================

    async sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        const imageInput = document.getElementById('imageInput');

        if (!message && (!imageInput || !imageInput.files.length)) {
            return;
        }

        // Add user message to chat
        this.addMessageToChat(message, 'user');

        // Clear input
        input.value = '';
        input.style.height = 'auto';

        // Show loading indicator
        const loader = document.getElementById('loadingIndicator');
        if (loader) loader.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('message', message);

            if (imageInput && imageInput.files[0]) {
                formData.append('image', imageInput.files[0]);
            }

            const response = await fetch('/chat', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (loader) loader.classList.add('hidden');

            if (data.success) {
                // Generate the table HTML directly
                const tableHtml = this.generateTableHtml(data);
                if (tableHtml) {
                    this.addMessageToChat(tableHtml, 'ai', true);
                } else {
                    this.addMessageToChat("No structured data could be extracted.", 'ai');
                }
            } else {
                this.addMessageToChat('Error: ' + (data.error || 'Unknown error'), 'ai');
            }

            // Clear image input
            if (imageInput) imageInput.value = '';
            const preview = document.getElementById('imagePreviewContainer');
            if (preview) preview.classList.add('hidden');

        } catch (error) {
            console.error('Chat error:', error);
            if (loader) loader.classList.add('hidden');
            this.addMessageToChat('Network error. Please try again.', 'ai');
        }
    } addMessageToChat(text, type, isHtml = false) {
        const container = document.getElementById('chatMessages');
        if (!container) return;

        const time = new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        const avatar = type === 'user'
            ? document.querySelector('.user-avatar')?.textContent || 'U'
            : '<i class="ph ph-robot"></i>';

        const sender = type === 'user' ? 'You' : 'AI Assistant';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-sender">${sender}</span>
                    <span class="message-time">${time}</span>
                </div>
                <div class="message-text">${isHtml ? text : this.escapeHtml(text)}</div>
            </div>
        `;

        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
    }

    generateTableHtml(data) {
        const tableData = data.table_data;
        this.currentTableData = tableData;

        if (!tableData || tableData.length === 0) {
            return null;
        }

        let actionsHtml = '<div class="export-actions">';
        actionsHtml += `<button onclick="app.downloadExcel()" class="btn btn-sm btn-excel"><i class="ph ph-file-xls"></i> Download Excel</button>`;
        actionsHtml += `<button onclick="app.downloadPpt()" class="btn btn-sm btn-ppt"><i class="ph ph-presentation"></i> Download PPT</button>`;
        actionsHtml += '</div>';

        // START TABLE
        let tableHtml = '<div class="table-responsive"><table class="data-table"><thead><tr>';

        tableHtml += `<th>Idea ID</th>`;
        tableHtml += `<th style="min-width: 250px;">Visual Scenarios</th>`; // Combined Column
        tableHtml += `<th>Action Strategy</th>`; // NEW
        tableHtml += `<th>Current Material/Spec</th>`; // NEW
        tableHtml += `<th>Proposed Material/Spec</th>`; // NEW
        tableHtml += `<th>Detailed Idea Description</th>`;
        tableHtml += `<th>Way Forward</th>`;
        tableHtml += `<th>Saving (INR)</th>`;
        tableHtml += `<th>Weight Saving</th>`;
        tableHtml += `<th>Status</th>`;
        tableHtml += `<th>Feasibility</th>`;
        tableHtml += `<th>Cost Saving</th>`;
        tableHtml += `<th>Weight Reduction</th>`;
        tableHtml += `<th>Homologation Feasibility</th>`;
        tableHtml += `<th>Homologation Theory</th>`;
        tableHtml += '</tr></thead><tbody>';

        tableData.forEach(row => {
            tableHtml += '<tr>';

            // ID
            tableHtml += `<td><span class="badge badge-gray">${this.escapeHtml(row['Idea Id'] || row['idea_id'])}</span></td>`;

            // IMAGES HELPER (Render Logic)
            const renderImg = (src, label) => {
                if (!src || src.includes('placeholder') || src === 'N/A' || src.includes('NaN')) {
                    return `
                        <div class="img-box">
                            <span class="img-label">${label}</span>
                            <div class="img-placeholder"><i class="ph ph-image"></i></div>
                        </div>`;
                }
                const cleanSrc = src.startsWith('http') ? src : `/${src.replace(/^\//, '')}`;
                return `
                    <div class="img-box">
                        <span class="img-label">${label}</span>
                        <img src="${cleanSrc}" 
                             onclick="window.open('${cleanSrc}', '_blank')" 
                             onerror="this.parentElement.innerHTML='<div class=\\'img-placeholder\\'><i class=\\'ph ph-warning\\'></i></div>'"
                             alt="${label}">
                    </div>`;
            };

            // Combined Image Cell (Competitor -> Current -> Proposal)
            const competitorImg = row['competitor_image'] || row['Competitor Image'];
            const currentImg = row['current_scenario_image'] || row['Current Scenario Image'];
            const proposalImg = row['proposal_scenario_image'] || row['Proposal Scenario Image'];

            tableHtml += `
                <td>
                    <div class="visual-scenario-container">
                        ${renderImg(competitorImg, "Competitor (KIA Seltos)")}
                        <div class="visual-arrow"><i class="ph ph-arrow-right"></i></div>
                        ${renderImg(currentImg, "Current MG")}
                        <div class="visual-arrow"><i class="ph ph-arrow-right"></i></div>
                        ${renderImg(proposalImg, "Implemented Overlay")}
                    </div>
                </td>`;

            // Data Columns
            // NEW SUMMARY COLUMNS
            const strategy = row['action_strategy'] || row['Action Strategy'] || '-';
            const curDesign = row['current_design'] || 'N/A';
            const propDesign = row['proposed_design'] || 'N/A';

            let strategyClass = 'badge-gray';
            if (strategy.includes('Material Substitution')) strategyClass = 'badge-info';
            if (strategy.includes('De-contenting')) strategyClass = 'badge-purple';
            if (strategy.includes('Part Consolidation')) strategyClass = 'badge-success';

            tableHtml += `<td><span class="badge ${strategyClass}">${this.escapeHtml(strategy)}</span></td>`;
            tableHtml += `<td><strong>${this.escapeHtml(curDesign)}</strong></td>`;
            tableHtml += `<td><strong style="color:var(--color-primary)">${this.escapeHtml(propDesign)}</strong></td>`;

            tableHtml += `<td class="text-wrap"><strong>${this.escapeHtml(row['Cost Reduction Idea'])}</strong><br><small class="text-muted">${this.escapeHtml(row['Origin'] || '')}</small></td>`;
            tableHtml += `<td class="text-wrap small-text">${this.escapeHtml(row['Way Forward'] || '-')}</td>`;
            tableHtml += `<td>${this.escapeHtml(row['Saving Value (INR)'] || '-')}</td>`;
            tableHtml += `<td>${this.escapeHtml(row['Weight Saving (Kg)'] || '-')}</td>`;

            // Status
            const status = row['Status'] || 'TBD';
            let badgeClass = 'role-User';
            if (status.includes('AI')) badgeClass = 'badge-purple';
            if (status === 'Web Sourced') badgeClass = 'badge-info';
            tableHtml += `<td><span class="role-badge ${badgeClass}">${status}</span></td>`;

            // Autonomous validation scores & homologation theory
            const feas = row['Feasibility Score'] ?? '-';
            const costScore = row['Cost Saving Score'] ?? '-';
            const weightScore = row['Weight Reduction Score'] ?? '-';
            const homoScore = row['Homologation Feasibility Score'] ?? '-';
            const homoTheory = row['Homologation Theory'] || '-';

            tableHtml += `<td>${this.escapeHtml(String(feas))}</td>`;
            tableHtml += `<td>${this.escapeHtml(String(costScore))}</td>`;
            tableHtml += `<td>${this.escapeHtml(String(weightScore))}</td>`;
            tableHtml += `<td>${this.escapeHtml(String(homoScore))}</td>`;
            tableHtml += `<td class="text-wrap small-text">${this.escapeHtml(homoTheory)}</td>`;

            tableHtml += '</tr>';
        });
        tableHtml += '</tbody></table></div>';
        return actionsHtml + tableHtml;
    }

    async downloadExcel() {
        if (!this.currentTableData || this.currentTableData.length === 0) {
            alert('No data available to export');
            return;
        }

        const btn = document.getElementById('downloadExcelBtn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<i class="ph ph-spinner"></i> Generating...';
        }

        try {
            const response = await fetch('/generate_excel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    table_data: this.currentTableData
                })
            });

            const result = await response.json();

            if (result.success && result.download_url) {
                // Trigger download
                window.location.href = result.download_url;
                this.showNotification('Excel file generated successfully', 'success');
            } else {
                throw new Error(result.error || 'Failed to generate Excel file');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showNotification('Failed to download Excel file', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<i class="ph ph-file-xls"></i> Download Excel';
            }
        }
    }

    async downloadPpt() {
        if (!this.currentTableData || this.currentTableData.length === 0) {
            alert('No data available to export');
            return;
        }

        const btn = document.getElementById('downloadPptBtn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<i class="ph ph-spinner"></i> Generating...';
        }

        try {
            const response = await fetch('/generate_ppt', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    table_data: this.currentTableData,
                    response_text: this.currentResponseText
                })
            });

            const result = await response.json();

            if (result.success && result.download_url) {
                // Trigger download
                window.location.href = result.download_url;
                this.showNotification('PPT file generated successfully', 'success');
            } else {
                throw new Error(result.error || 'Failed to generate PPT');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showNotification('Failed to download PPT', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<i class="ph ph-presentation"></i> Download PPT';
            }
        }
    }

    showNotification(message, type = 'info') {
        // Simple notification - you can enhance this with your existing notification system
        const notification = document.createElement('div');
        notification.className = `notification ${type} `;
        notification.textContent = message;
        notification.style.cssText = `
position: fixed;
top: 20px;
right: 20px;
padding: 12px 20px;
background: ${type === 'success' ? '#4caf50' : type === 'error' ? '#f44336' : '#2196f3'};
color: white;
border - radius: 4px;
z - index: 10000;
box - shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
`;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    // ========================================
    // History
    // ========================================

    async loadHistory() {
        const container = document.getElementById('historyContainer');
        if (!container) return;

        container.innerHTML = '<p class="empty-text">Loading history...</p>';

        try {
            const response = await fetch('/history');
            const data = await response.json();

            // Store in class instance for later retrieval
            this.historyCache = data.history || [];

            if (this.historyCache.length > 0) {
                let html = '<div class="history-grid">';

                this.historyCache.forEach((item, index) => {
                    const dateObj = new Date(item.timestamp);
                    const dateStr = dateObj.toLocaleDateString() + ' ' + dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

                    html += `
                        <div class="history-card">
                            <div class="history-card-header">
                                <span class="history-time"><i class="ph ph-clock"></i> ${dateStr}</span>
                            </div>
                            <div class="history-card-body">
                                <div class="history-query"><strong>Q:</strong> ${this.escapeHtml(item.query || 'No query')}</div>
                                <div class="history-response">${this.escapeHtml(item.response || 'No text response.')}</div>
                            </div>
                            <div class="history-card-footer">
                                ${item.has_table ? `<button class="btn btn-primary btn-sm" onclick="app.viewHistoricalResults(${index})"><i class="ph ph-table"></i> View Data Table</button>` : '<span class="text-muted">No Table Data</span>'}
                            </div>
                        </div>
                    `;
                });

                html += '</div>';
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p class="empty-text">No history found.</p>';
            }
        } catch (error) {
            console.error('History error:', error);
            container.innerHTML = '<p class="empty-text" style="color: var(--color-danger);">Error loading history</p>';
        }
    }

    viewHistoricalResults(index) {
        const item = this.historyCache[index];
        if (!item || !item.has_table) return;

        // Switch to Chat Tab
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        document.querySelector('.nav-item[data-section="chat"]').classList.add('active');

        document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
        document.getElementById('chatSection').classList.add('active');

        // Populate Chat Feed with Historical Anchor
        this.addMessageToChat(item.query, 'user');

        // Generate and Inject the Table HTML
        this.currentTableData = item.table_data;
        this.currentResponseText = item.response || "Historical table loaded.";
        const tableHtml = this.generateTableHtml({ table_data: item.table_data, response_text: item.response });

        this.addMessageToChat(tableHtml, 'ai', true);
    }

    // ========================================
    // Database Management
    // ========================================

    async handleDatabaseUpload(e) {
        e.preventDefault();

        const excelInput = document.getElementById('databaseExcelFile');
        const zipInput = document.getElementById('databaseZipFile');
        const uploadBtn = document.getElementById('dbUploadBtn');
        const progressDiv = document.getElementById('dbUploadProgress');
        const progressFill = document.getElementById('dbProgressFill');
        const progressText = document.getElementById('dbProgressText');

        if (!excelInput || !excelInput.files[0]) {
            this.showFlash('Please select an Excel file', 'error');
            return;
        }
        if (!zipInput || !zipInput.files[0]) {
            this.showFlash('Please select an Images ZIP file', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('excel_file', excelInput.files[0]);
        formData.append('zip_file', zipInput.files[0]);

        // Show progress UI
        if (uploadBtn) { uploadBtn.disabled = true; uploadBtn.innerHTML = '<i class="ph ph-circle-notch"></i> Uploading...'; }
        if (progressDiv) progressDiv.classList.remove('hidden');
        if (progressFill) progressFill.style.width = '30%';
        if (progressText) progressText.textContent = 'Uploading files...';

        try {
            if (progressFill) progressFill.style.width = '60%';
            if (progressText) progressText.textContent = 'Processing & rebuilding index...';

            const response = await fetch('/upload_database', {
                method: 'POST',
                body: formData
            });

            if (progressFill) progressFill.style.width = '90%';

            const data = await response.json();

            if (data.success) {
                if (progressFill) progressFill.style.width = '100%';
                if (progressText) progressText.textContent = 'Complete!';
                setTimeout(() => { if (progressDiv) progressDiv.classList.add('hidden'); }, 2000);

                this.showFlash(data.message || 'Database uploaded successfully', 'success');
                excelInput.value = '';
                zipInput.value = '';
                document.getElementById('databaseExcelFileName').textContent = 'No file selected';
                document.getElementById('databaseZipFileName').textContent = 'No file selected';
                document.getElementById('excelDropzone')?.classList.remove('has-file');
                document.getElementById('zipDropzone')?.classList.remove('has-file');
                this.loadDatabaseStatus();
                this.loadDatabaseUploads();
            } else {
                if (progressDiv) progressDiv.classList.add('hidden');
                this.showFlash(data.error || 'Upload failed', 'error');
            }
        } catch (error) {
            console.error('Upload error:', error);
            if (progressDiv) progressDiv.classList.add('hidden');
            this.showFlash('Network error during upload', 'error');
        } finally {
            if (uploadBtn) { uploadBtn.disabled = false; uploadBtn.innerHTML = '<i class="ph ph-cloud-arrow-up"></i> Upload & Rebuild Knowledge Base'; }
        }
    }

    async loadDatabaseStatus() {
        try {
            const response = await fetch('/database_status');
            const data = await response.json();

            if (data.success) {
                document.getElementById('totalRecords').textContent = data.total_records || '-';
                document.getElementById('lastUpdated').textContent = data.last_updated || '-';
                document.getElementById('dbSize').textContent = data.size || '-';
            }
        } catch (error) {
            console.error('Database status error:', error);
        }
    }

    async loadDatabaseUploads() {
        const body = document.getElementById('databaseUploadsBody');
        if (!body) return;

        body.innerHTML = `<tr><td colspan="6" class="db-empty-row"><i class="ph ph-spinner"></i><span>Loading...</span></td></tr>`;

        try {
            const res = await fetch('/database_uploads');
            const data = await res.json();
            if (!data.success) {
                body.innerHTML = `<tr><td colspan="6" class="db-empty-row"><i class="ph ph-warning"></i><span>${this.escapeHtml(data.error || 'Failed to load uploads')}</span></td></tr>`;
                return;
            }

            const uploads = data.uploads || [];
            const countEl = document.getElementById('uploadHistoryCount');
            if (countEl) countEl.textContent = uploads.length;

            if (uploads.length === 0) {
                body.innerHTML = `<tr><td colspan="6" class="db-empty-row"><i class="ph ph-clock-countdown"></i><span>No uploads yet</span></td></tr>`;
                return;
            }

            let html = '';
            uploads.forEach(u => {
                const date = u.created_at ? new Date(u.created_at).toLocaleString() : '—';
                const status = (u.status || 'unknown').toLowerCase();
                let badgeClass = 'db-status-badge--pending';
                let badgeIcon = 'ph-clock';
                if (status === 'success') { badgeClass = 'db-status-badge--success'; badgeIcon = 'ph-check-circle'; }
                else if (status === 'error' || status === 'failed') { badgeClass = 'db-status-badge--error'; badgeIcon = 'ph-x-circle'; }

                html += `<tr>
                    <td>${this.escapeHtml(date)}</td>
                    <td>
                        <div class="db-file-cell">
                            <i class="ph ph-microsoft-excel-logo db-file-cell-icon" style="color:#16a34a"></i>
                            <span title="${this.escapeHtml(u.excel_filename || '')}">"${this.escapeHtml(u.excel_filename || '—')}</span>
                        </div>
                    </td>
                    <td>
                        <div class="db-file-cell">
                            <i class="ph ph-file-zip db-file-cell-icon" style="color:#d97706"></i>
                            <span title="${this.escapeHtml(u.zip_filename || '')}">"${this.escapeHtml(u.zip_filename || '—')}</span>
                        </div>
                    </td>
                    <td>
                        <div class="user-cell">
                            <div class="user-cell-avatar" style="width:24px;height:24px;font-size:11px">${this.escapeHtml((u.uploaded_by || 'U')[0].toUpperCase())}</div>
                            <span>${this.escapeHtml(u.uploaded_by || '—')}</span>
                        </div>
                    </td>
                    <td>${u.records_count ? u.records_count : '—'}</td>
                    <td><span class="db-status-badge ${badgeClass}"><i class="ph ${badgeIcon}"></i>${u.status || 'unknown'}</span></td>
                </tr>`;
            });
            body.innerHTML = html;
        } catch (err) {
            console.error('Database uploads error:', err);
            body.innerHTML = `<tr><td colspan="6" class="db-empty-row"><i class="ph ph-warning"></i><span>Error loading uploads</span></td></tr>`;
        }
    }
    // ========================================
    // User Management
    // ========================================

    toggleAddUserForm() {
        const form = document.getElementById('addUserForm');
        if (form) {
            form.classList.toggle('hidden');
        }
    }

    async handleCreateUser(e) {
        e.preventDefault();

        const formData = new FormData(e.target);
        const data = {
            username: formData.get('username'),
            password: formData.get('password'),
            role: formData.get('role')
        };

        try {
            const response = await fetch('/auth/create_user', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.showFlash('User created successfully', 'success');
                e.target.reset();
                this.toggleAddUserForm();
                setTimeout(() => window.location.reload(), 1000);
            } else {
                this.showFlash(result.error || 'Failed to create user', 'error');
            }
        } catch (error) {
            console.error('Create user error:', error);
            this.showFlash('Network error', 'error');
        }
    }

    async resetPassword(username) {
        if (!confirm(`Reset password for user: ${username}?`)) {
            return;
        }

        try {
            const response = await fetch('/auth/reset_password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username })
            });

            const data = await response.json();

            if (data.success) {
                this.showFlash(`Password reset.New password: ${data.new_password} `, 'success');
            } else {
                this.showFlash(data.error || 'Failed to reset password', 'error');
            }
        } catch (error) {
            console.error('Reset password error:', error);
            this.showFlash('Network error', 'error');
        }
    }

    async deleteUser(username) {
        if (!confirm(`Delete user: ${username}? This action cannot be undone.`)) {
            return;
        }

        try {
            const response = await fetch('/auth/delete_user', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username })
            });

            const data = await response.json();

            if (data.success) {
                this.showFlash('User deleted successfully', 'success');
                setTimeout(() => window.location.reload(), 1000);
            } else {
                this.showFlash(data.error || 'Failed to delete user', 'error');
            }
        } catch (error) {
            console.error('Delete user error:', error);
            this.showFlash('Network error', 'error');
        }
    }

    // ========================================
    // System Logs
    // ========================================

    async loadLogs() {
        const container = document.getElementById('logsContainer');
        if (!container) return;

        container.innerHTML = '<p class="empty-text">Loading logs...</p>';

        try {
            const response = await fetch('/system_logs');
            const data = await response.json();

            if (data.success && data.logs) {
                container.innerHTML = `< pre > ${this.escapeHtml(data.logs)}</pre > `;
            } else {
                container.innerHTML = '<p class="empty-text">No logs available</p>';
            }
        } catch (error) {
            console.error('Logs error:', error);
            container.innerHTML = '<p class="empty-text" style="color: var(--color-danger);">Error loading logs</p>';
        }
    }

    // ========================================
    // Settings
    // ========================================

    async handlePasswordChange(e) {
        e.preventDefault();

        const formData = new FormData(e.target);
        const data = {
            current_password: formData.get('current_password'),
            new_password: formData.get('new_password'),
            confirm_password: formData.get('confirm_password')
        };

        if (data.new_password !== data.confirm_password) {
            this.showFlash('New passwords do not match', 'error');
            return;
        }

        try {
            const response = await fetch('/auth/change_password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.showFlash('Password changed successfully', 'success');
                e.target.reset();
            } else {
                this.showFlash(result.error || 'Failed to change password', 'error');
            }
        } catch (error) {
            console.error('Password change error:', error);
            this.showFlash('Network error', 'error');
        }
    }

    // ========================================
    // Big Data / Analytics
    // ========================================

    async loadAnalytics() {
        const totalEl = document.getElementById('bdTotalIdeas');
        const deptCountEl = document.getElementById('bdDeptCount');
        const deptChartEl = document.getElementById('bdDeptChart');
        const topSavingsEl = document.getElementById('bdTopSavings');

        if (deptChartEl) {
            deptChartEl.innerHTML = 'Loading...';
        }

        try {
            // 1) Query analytical aggregates over ideas (semantic layer)
            const resAgg = await fetch('/analytics/ideas_summary');
            const agg = await resAgg.json();

            // 2) Query lake-wide load stats (physical data layout)
            const resLake = await fetch('/analytics/lake_status');
            const lake = await resLake.json();

            if (!agg.success) {
                if (deptChartEl) deptChartEl.innerHTML = this.escapeHtml(agg.error || 'Failed to load analytics');
                return;
            }
            if (!lake.success) {
                if (deptChartEl) deptChartEl.innerHTML = this.escapeHtml(lake.error || 'Failed to load lake status');
                return;
            }

            const stats = agg.data || {};
            const lakeStats = lake.data || {};

            if (totalEl) totalEl.textContent = stats.total_ideas ?? '-';
            if (deptCountEl && Array.isArray(stats.by_dept)) {
                deptCountEl.textContent = `${stats.by_dept.length} depts / ${lakeStats.load_count || 0} loads`;
            }

            // Simple bar chart for departments
            if (deptChartEl) {
                const byDept = stats.by_dept || [];
                if (byDept.length === 0) {
                    deptChartEl.innerHTML = '<p class="empty-text">No department data in data lake.</p>';
                } else {
                    const maxCount = Math.max(...byDept.map(d => d.idea_count || 0), 1);
                    let html = '<div class="bar-chart">';
                    byDept.slice(0, 10).forEach(d => {
                        const width = (100 * (d.idea_count || 0)) / maxCount;
                        html += `
    < div class="bar-row" >
                                <span class="bar-label">${this.escapeHtml(d.dept || 'Unknown')}</span>
                                <div class="bar-track">
                                    <div class="bar-fill" style="width:${width}%"></div>
                                </div>
                                <span class="bar-value">${d.idea_count}</span>
                            </div >
    `;
                    });
                    html += '</div>';

                    // Show load timeline if available
                    const byDate = lakeStats.by_date || [];
                    if (byDate.length > 0) {
                        html += '<div class="chart-subtitle">Rows per load_date:</div>';
                        html += '<div class="bar-chart small">';
                        const maxRows = Math.max(...byDate.map(d => d.rows || 0), 1);
                        byDate.forEach(d => {
                            const w = (100 * (d.rows || 0)) / maxRows;
                            html += `
    < div class="bar-row" >
                                    <span class="bar-label">${this.escapeHtml(d.load_date || '')}</span>
                                    <div class="bar-track">
                                        <div class="bar-fill" style="width:${w}%"></div>
                                    </div>
                                    <span class="bar-value">${d.rows}</span>
                                </div >
    `;
                        });
                        html += '</div>';
                    }

                    deptChartEl.innerHTML = html;
                }
            }

            // Top savings list
            if (topSavingsEl) {
                const list = stats.top_savings || [];
                if (list.length === 0) {
                    topSavingsEl.innerHTML = '<p class="empty-text">No saving data available.</p>';
                } else {
                    let html = '';
                    list.forEach(item => {
                        html += `
    < div class="activity-item" >
                                <div class="activity-icon">
                                    <i class="ph ph-currency-inr"></i>
                                </div>
                                <div class="activity-content">
                                    <div class="activity-text">
                                        <strong>${this.escapeHtml(item.idea_id || '')}</strong> - 
                                        ${this.escapeHtml(item.cost_reduction_idea || '')}
                                    </div>
                                    <div class="activity-time">
                                        Saving: ${this.escapeHtml(String(item.saving_value_inr ?? 'N/A'))}
                                    </div>
                                </div>
                            </div >
    `;
                    });
                    topSavingsEl.innerHTML = html;
                }
            }
        } catch (err) {
            console.error('Analytics load error:', err);
        }
    }

    // ========================================
    // Big Data Dashboard — Full Controller
    // ========================================

    /** Cache for dashboard data so charts can redraw without refetching */
    _bdData = null;
    _bdLake = null;
    _charts = {};   // Keyed by canvas id
    _leaderSort = { col: 'saving', dir: 'desc' };
    _leaderAllRows = [];

    async loadAnalytics() {
        const refreshEl = document.getElementById('bdLastRefresh');
        if (refreshEl) refreshEl.textContent = 'Loading…';

        try {
            const [resDetail, resLake] = await Promise.all([
                fetch('/analytics/ideas_detail'),
                fetch('/analytics/lake_status'),
            ]);
            const detail = await resDetail.json();
            const lakeRes = await resLake.json();

            if (!detail.success) throw new Error(detail.error || 'Detail fetch failed');

            this._bdData = detail.data;
            this._bdLake = lakeRes.success ? lakeRes.data : {};

            this._renderKpi(this._bdData.kpi, this._bdData.ai_generated_count);
            this._renderStatusDonut(this._bdData.by_status);
            this._renderDeptBar(this._bdData.by_dept, document.getElementById('bdDeptMetric')?.value || 'saving');
            this._renderComponentBar(this._bdData.by_component, 'total_saving');
            this._renderScatter(this._bdData.scatter, 'dept');
            this._renderTimeline(this._bdLake);
            this._renderFeasibility(this._bdData.feasibility_matrix);
            this._renderLeaderboard(this._bdData.top_ideas);
            this._renderSourceBreakdown(this._bdData.by_source, this._bdData.ai_generated_count);

            if (refreshEl) refreshEl.textContent = 'Refreshed ' + new Date().toLocaleTimeString();
        } catch (err) {
            console.error('Big Data dashboard error:', err);
            if (refreshEl) refreshEl.textContent = 'Error loading data';
        }
    }

    _palette(n) {
        const base = ['#e53935', '#8e24aa', '#1e88e5', '#43a047', '#fb8c00', '#00acc1', '#6d4c41', '#5e35b1', '#3949ab', '#00897b', '#c0ca33', '#f4511e'];
        return Array.from({ length: n }, (_, i) => base[i % base.length]);
    }

    _destroyChart(id) {
        if (this._charts[id]) { this._charts[id].destroy(); delete this._charts[id]; }
    }

    /** Resolve the actual hex/rgb value of a CSS variable for Chart.js (which can't parse var()) */
    _tickColor() {
        return getComputedStyle(document.documentElement).getPropertyValue('--color-text-secondary').trim() || '#666';
    }
    _gridColor() { return 'rgba(128,128,128,0.12)'; }

    _renderKpi(kpi, aiGenCount) {
        if (!kpi) return;
        const fmt = n => n >= 1e7 ? (n / 1e7).toFixed(1) + 'Cr' : n >= 1e5 ? (n / 1e5).toFixed(1) + 'L' : Math.round(n).toLocaleString();
        const el = id => document.getElementById(id);
        if (el('bdKpiTotal')) el('bdKpiTotal').textContent = (kpi.total_ideas || 0).toLocaleString();
        if (el('bdKpiSaving')) el('bdKpiSaving').textContent = '₹' + fmt(kpi.total_inr_saving || 0);
        if (el('bdKpiWeight')) el('bdKpiWeight').textContent = (kpi.avg_weight_saving || 0).toFixed(2);
        if (el('bdKpiDepts')) el('bdKpiDepts').textContent = kpi.dept_count || 0;
        if (el('bdKpiCapex')) el('bdKpiCapex').textContent = '₹' + fmt(kpi.total_capex || 0);
        if (el('bdKpiAiGen')) el('bdKpiAiGen').textContent = (aiGenCount || 0).toLocaleString();
    }

    _renderStatusDonut(byStatus) {
        const canvas = document.getElementById('bdStatusChart');
        if (!canvas || !byStatus?.length) return;
        this._destroyChart('bdStatusChart');

        const labels = byStatus.map(d => d.status);
        const values = byStatus.map(d => d.idea_count);
        const colors = this._palette(labels.length);

        this._charts['bdStatusChart'] = new Chart(canvas, {
            type: 'doughnut',
            data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 2, borderColor: 'var(--color-surface)' }] },
            options: {
                responsive: true, maintainAspectRatio: false, cutout: '62%',
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed} ideas` } },
                },
            }
        });

        // Build legend
        const legend = document.getElementById('bdStatusLegend');
        if (legend) {
            legend.innerHTML = labels.map((l, i) => `
                <div class="bd-legend-row">
                    <span class="bd-legend-dot" style="background:${colors[i]}"></span>
                    <span class="bd-legend-label">${this.escapeHtml(l)}</span>
                    <span class="bd-legend-val">${values[i]}</span>
                </div>`).join('');
        }
    }

    _renderDeptBar(byDept, metric = 'saving') {
        const canvas = document.getElementById('bdDeptBarChart');
        if (!canvas || !byDept?.length) return;
        this._destroyChart('bdDeptBarChart');

        const labels = byDept.map(d => d.dept);
        const values = byDept.map(d => metric === 'count' ? d.idea_count : metric === 'avg' ? d.avg_saving : d.total_saving);
        const axisLabel = metric === 'count' ? 'Idea Count' : 'INR';

        this._charts['bdDeptBarChart'] = new Chart(canvas, {
            type: 'bar',
            data: { labels, datasets: [{ label: axisLabel, data: values, backgroundColor: this._palette(labels.length), borderRadius: 6 }] },
            options: {
                indexAxis: 'y',
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: this._gridColor() }, ticks: { color: this._tickColor() } },
                    y: { grid: { display: false }, ticks: { color: this._tickColor(), font: { size: 11 } } },
                },
            }
        });
    }

    updateDeptChart() {
        if (!this._bdData) return;
        const metric = document.getElementById('bdDeptMetric')?.value || 'saving';
        this._renderDeptBar(this._bdData.by_dept, metric);
    }

    _renderComponentBar(byComponent, metric = 'total_saving') {
        const canvas = document.getElementById('bdComponentChart');
        if (!canvas || !byComponent?.length) return;
        this._destroyChart('bdComponentChart');

        const labels = byComponent.map(d => d.component);
        const values = byComponent.map(d => d[metric] || 0);

        this._charts['bdComponentChart'] = new Chart(canvas, {
            type: 'bar',
            data: { labels, datasets: [{ label: metric, data: values, backgroundColor: '#e53935cc', borderColor: '#e53935', borderWidth: 1, borderRadius: 4 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: this._tickColor(), font: { size: 10 }, maxRotation: 40 } },
                    y: { grid: { color: this._gridColor() }, ticks: { color: this._tickColor() } },
                },
            }
        });
    }

    switchComponentMetric(btn, metric) {
        document.querySelectorAll('.bd-tab').forEach(b => b.classList.remove('bd-tab--active'));
        btn.classList.add('bd-tab--active');
        if (this._bdData) this._renderComponentBar(this._bdData.by_component, metric);
    }

    _renderScatter(scatter, colorBy = 'dept') {
        const canvas = document.getElementById('bdScatterChart');
        if (!canvas || !scatter?.length) return;
        this._destroyChart('bdScatterChart');

        // Group by colorBy dimension
        const groups = {};
        scatter.forEach(pt => {
            const key = colorBy === 'dept' ? (pt.dept || '?') : (pt.status || '?');
            if (!groups[key]) groups[key] = [];
            groups[key].push({ x: pt.x, y: pt.y, idea_id: pt.idea_id });
        });
        const colors = this._palette(Object.keys(groups).length);

        const datasets = Object.keys(groups).map((key, i) => ({
            label: key,
            data: groups[key],
            backgroundColor: colors[i] + 'b0',
            pointRadius: 6, pointHoverRadius: 9,
        }));

        this._charts['bdScatterChart'] = new Chart(canvas, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { color: this._tickColor(), boxWidth: 10, font: { size: 10 } } },
                    tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ₹${ctx.parsed.x} | ${ctx.parsed.y}kg` } },
                },
                scales: {
                    x: { title: { display: true, text: 'Cost Saving (INR)', color: this._tickColor() }, grid: { color: this._gridColor() }, ticks: { color: this._tickColor() } },
                    y: { title: { display: true, text: 'Weight Saving (kg)', color: this._tickColor() }, grid: { color: this._gridColor() }, ticks: { color: this._tickColor() } },
                },
            }
        });
    }

    updateScatterColor() {
        if (!this._bdData) return;
        this._renderScatter(this._bdData.scatter, document.getElementById('bdScatterColor')?.value || 'dept');
    }

    _renderTimeline(lake) {
        const canvas = document.getElementById('bdTimelineChart');
        if (!canvas) return;
        this._destroyChart('bdTimelineChart');

        const byDate = lake?.by_date || [];
        if (byDate.length === 0) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = this._tickColor();
            ctx.font = '13px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No load history yet', canvas.offsetWidth / 2, 100);
            return;
        }

        const labels = byDate.map(d => d.load_date);
        const values = byDate.map(d => d.rows);

        this._charts['bdTimelineChart'] = new Chart(canvas, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Rows Loaded', data: values,
                    borderColor: '#e53935', backgroundColor: 'rgba(229,57,53,0.12)',
                    tension: 0.4, fill: true, pointRadius: 6, pointBackgroundColor: '#e53935',
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: this._tickColor() } },
                    y: { grid: { color: this._gridColor() }, ticks: { color: this._tickColor() }, beginAtZero: true },
                },
            }
        });
    }

    _renderSourceBreakdown(bySource, aiGenCount) {
        const el = document.getElementById('bdSourceBreakdown');
        if (!el) return;
        if (!bySource?.length) { el.textContent = 'No source data'; return; }

        const sourceColors = { 'ASI': '#1e88e5', 'MGI': '#43a047', 'Unknown': '#9e9e9e' };
        const total = bySource.reduce((s, r) => s + r.idea_count, 0);

        el.innerHTML = `
            <div class="bd-source-grid">
                ${bySource.map(r => {
            const pct = total ? ((r.idea_count / total) * 100).toFixed(1) : 0;
            const color = sourceColors[r.source] || '#e53935';
            return `<div class="bd-source-row">
                        <span class="bd-source-pill" style="background:${color}20;color:${color};border:1px solid ${color}40">${this.escapeHtml(r.source)}</span>
                        <div class="bd-source-bar-wrap">
                            <div class="bd-source-bar" style="width:${pct}%;background:${color}"></div>
                        </div>
                        <span class="bd-source-count">${r.idea_count} <small>(${pct}%)</small></span>
                    </div>`;
        }).join('')}
                ${aiGenCount > 0 ? `<div class="bd-source-row bd-source-ai">
                    <span class="bd-source-pill bd-source-pill--ai"><i class="ph ph-robot"></i> AI Chat</span>
                    <div class="bd-source-bar-wrap"><div class="bd-source-bar" style="width:100%;background:#8e24aa"></div></div>
                    <span class="bd-source-count">${aiGenCount} <small>sessions with tables</small></span>
                </div>` : ''}
            </div>`;
    }

    _renderFeasibility(matrix) {
        const grid = document.getElementById('bdFeasibilityGrid');
        if (!grid || !matrix?.length) { if (grid) grid.textContent = 'No feasibility data'; return; }

        const peVals = [...new Set(matrix.map(m => m.pe))].sort();
        const finVals = [...new Set(matrix.map(m => m.fin))].sort();
        const lookup = {};
        matrix.forEach(m => { lookup[`${m.pe}::${m.fin}`] = m.count; });
        const maxCount = Math.max(...matrix.map(m => m.count), 1);

        let html = `<div class="bd-feas-table">`;
        html += `<div class="bd-feas-corner"></div>`;
        finVals.forEach(f => { html += `<div class="bd-feas-col-hdr" title="Financial: ${f}">${this.escapeHtml(f.substring(0, 8))}</div>`; });
        peVals.forEach(pe => {
            html += `<div class="bd-feas-row-hdr" title="PE: ${pe}">${this.escapeHtml(pe.substring(0, 8))}</div>`;
            finVals.forEach(fin => {
                const cnt = lookup[`${pe}::${fin}`] || 0;
                const intensity = Math.round(40 + (cnt / maxCount) * 200);
                const bg = cnt ? `rgba(229,57,53,${(cnt / maxCount).toFixed(2)})` : 'var(--color-bg)';
                const color = cnt / maxCount > 0.5 ? '#fff' : 'var(--color-text)';
                html += `<div class="bd-feas-cell" style="background:${bg};color:${color}" title="${pe} × ${fin}: ${cnt}">${cnt || ''}</div>`;
            });
        });
        html += `</div>`;
        html += `<p class="bd-feas-hint"><span style="background:rgba(229,57,53,0.15);padding:2px 6px;border-radius:4px">Lighter</span> = fewer ideas &nbsp; <span style="background:rgba(229,57,53,0.8);color:#fff;padding:2px 6px;border-radius:4px">Darker</span> = more ideas</p>`;
        grid.innerHTML = html;
    }

    _renderLeaderboard(ideas) {
        this._leaderAllRows = ideas || [];
        this._drawLeaderboard(ideas);
    }

    _drawLeaderboard(rows) {
        const tbody = document.getElementById('bdLeaderboardBody');
        if (!tbody) return;

        if (!rows?.length) {
            tbody.innerHTML = `<tr><td colspan="7" class="db-empty-row"><i class="ph ph-empty"></i><span>No ideas found</span></td></tr>`;
            return;
        }

        const statusColors = {
            'AI Generated': 'badge-purple', 'Web Sourced': 'badge-info',
            'Existing': 'badge-success', 'TBD': 'badge-gray', 'Unknown': 'badge-gray',
        };

        tbody.innerHTML = rows.map((r, i) => {
            const sc = statusColors[r.status] || 'badge-gray';
            return `<tr class="${i % 2 === 0 ? 'bd-row-even' : ''}">
                <td><span class="badge badge-gray">${this.escapeHtml(r.idea_id || '')}</span></td>
                <td class="bd-title-cell" title="${this.escapeHtml(r.title)}">${this.escapeHtml((r.title || '').substring(0, 80))}${r.title?.length > 80 ? '…' : ''}</td>
                <td><span class="bd-dept-pill">${this.escapeHtml(r.dept || '?')}</span></td>
                <td><span class="badge ${sc}">${this.escapeHtml(r.status || 'TBD')}</span></td>
                <td class="bd-num-cell">₹${(r.saving_inr || 0).toLocaleString()}</td>
                <td class="bd-num-cell">${(r.weight_saving || 0).toFixed(2)}</td>
                <td class="bd-num-cell">${r.capex ? '₹' + (r.capex).toLocaleString() : '—'}</td>
            </tr>`;
        }).join('');
    }

    filterLeaderboard() {
        const q = (document.getElementById('bdLeaderSearch')?.value || '').toLowerCase();
        if (!q) { this._drawLeaderboard(this._leaderAllRows); return; }
        const filtered = this._leaderAllRows.filter(r =>
            (r.idea_id || '').toLowerCase().includes(q) ||
            (r.title || '').toLowerCase().includes(q) ||
            (r.dept || '').toLowerCase().includes(q) ||
            (r.status || '').toLowerCase().includes(q)
        );
        this._drawLeaderboard(filtered);
    }

    sortLeaderboard(col) {
        if (this._leaderSort.col === col) {
            this._leaderSort.dir = this._leaderSort.dir === 'asc' ? 'desc' : 'asc';
        } else {
            this._leaderSort = { col, dir: 'desc' };
        }
        // Update header indicators
        document.querySelectorAll('.bd-sortable').forEach(th => th.classList.remove('bd-sort-active', 'bd-sort-asc', 'bd-sort-desc'));
        const activeTh = document.querySelector(`.bd-sortable[data-sort="${col}"]`);
        if (activeTh) activeTh.classList.add('bd-sort-active', `bd-sort-${this._leaderSort.dir}`);

        // Sort locally
        const sorted = [...this._leaderAllRows].sort((a, b) => {
            const keyMap = { saving: 'saving_inr', weight: 'weight_saving', capex: 'capex', dept: 'dept', idea_id: 'idea_id', status: 'status' };
            const k = keyMap[col] || col;
            const av = a[k] ?? '', bv = b[k] ?? '';
            if (typeof av === 'number') return this._leaderSort.dir === 'asc' ? av - bv : bv - av;
            return this._leaderSort.dir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
        });
        this._drawLeaderboard(sorted);
    }

    async loadFullTable() {
        if (!this._bdData) { await this.loadAnalytics(); return; }
        const { col, dir } = this._leaderSort;
        const res = await fetch(`/analytics/ideas_table?sort=${col}&order=${dir}&limit=200`);
        const d = await res.json();
        if (d.success) {
            this._leaderAllRows = d.data.map(r => ({
                idea_id: r.idea_id, title: r.title, dept: r.dept, status: r.status,
                saving_inr: r.saving_inr, weight_saving: r.weight_saving, capex: r.capex,
            }));
            this._drawLeaderboard(this._leaderAllRows);
        }
    }

    // Configure panel
    openConfigPanel() {
        document.getElementById('bdConfigPanel')?.classList.add('bd-configure-panel--open');
        document.getElementById('bdConfigOverlay')?.classList.add('bd-configure-overlay--visible');
        // Wire up toggles
        document.querySelectorAll('.bd-widget-toggle').forEach(cb => {
            cb.onchange = () => {
                const w = document.getElementById(cb.dataset.widget);
                if (w) w.style.display = cb.checked ? '' : 'none';
            };
        });
    }

    closeConfigPanel() {
        document.getElementById('bdConfigPanel')?.classList.remove('bd-configure-panel--open');
        document.getElementById('bdConfigOverlay')?.classList.remove('bd-configure-overlay--visible');
    }

    // ========================================
    // Utilities
    // ========================================

    escapeHtml(text) {
        if (typeof text !== 'string') return text;
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showFlash(message, category = 'info') {
        let flashContainer = document.querySelector('.flash-messages');

        if (!flashContainer) {
            flashContainer = document.createElement('div');
            flashContainer.className = 'flash-messages';
            document.body.appendChild(flashContainer);
        }

        const iconMap = {
            success: 'check-circle',
            error: 'x-circle',
            info: 'info'
        };

        const flashDiv = document.createElement('div');
        flashDiv.className = `flash-message flash-${category}`;
        flashDiv.innerHTML = `
    <i class="ph ph-${iconMap[category] || 'info'}"></i>
            <span>${this.escapeHtml(message)}</span>
            <button class="flash-close">
                <i class="ph ph-x"></i>
            </button>
`;

        flashContainer.appendChild(flashDiv);

        // Close button
        flashDiv.querySelector('.flash-close').addEventListener('click', () => {
            flashDiv.remove();
        });

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (flashDiv.parentElement) {
                flashDiv.remove();
            }
        }, 5000);
    }
}


// ========================================
// Initialize App
// ========================================

let app;

document.addEventListener('DOMContentLoaded', () => {
    app = new ChatbotApp();
});

// Export for global access
window.app = app;