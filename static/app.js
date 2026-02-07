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

        // Database file inputs
        const databaseExcelFile = document.getElementById('databaseExcelFile');
        if (databaseExcelFile) {
            databaseExcelFile.addEventListener('change', (e) => {
                const fileName = e.target.files[0]?.name || 'No file chosen';
                document.getElementById('databaseExcelFileName').textContent = fileName;
            });
        }

        const databaseZipFile = document.getElementById('databaseZipFile');
        if (databaseZipFile) {
            databaseZipFile.addEventListener('change', (e) => {
                const fileName = e.target.files[0]?.name || 'No file chosen';
                document.getElementById('databaseZipFileName').textContent = fileName;
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
                this.addMessageToChat(data.response_text, 'ai');
                this.renderResults(data);
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
    }

    addMessageToChat(text, type) {
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
                <div class="message-text">${this.escapeHtml(text)}</div>
            </div>
        `;
        
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
    }

renderResults(data) {
        const resultsContainer = document.getElementById('resultsContent');
        if (!resultsContainer) return;

        const tableData = data.table_data;
        const responseText = data.response_text || '';
        this.currentTableData = tableData;
        this.currentResponseText = responseText;

        if (!tableData || tableData.length === 0) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <i class="ph ph-info"></i>
                    <p>No structured data available for this query.</p>
                </div>
            `;
            return;
        }

        let actionsHtml = '<div class="export-actions">';
        actionsHtml += `<button id="downloadExcelBtn" class="btn btn-sm btn-excel"><i class="ph ph-file-xls"></i> Download Excel</button>`;
        actionsHtml += `<button id="downloadPptBtn" class="btn btn-sm btn-ppt"><i class="ph ph-presentation"></i> Download PPT</button>`;
        actionsHtml += '</div>';

        // START TABLE
        // START TABLE
        let tableHtml = '<div class="table-responsive"><table class="data-table"><thead><tr>';
        
        tableHtml += `<th>Idea ID</th>`;
        tableHtml += `<th style="min-width: 250px;">Visual Scenarios</th>`; // Combined Column
        tableHtml += `<th>Cost Reduction Idea</th>`;
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
            tableHtml += `<td class="text-wrap"><strong>${this.escapeHtml(row['Cost Reduction Idea'])}</strong><br><small class="text-muted">${this.escapeHtml(row['Origin'])}</small></td>`;
            tableHtml += `<td class="text-wrap small-text">${this.escapeHtml(row['Way Forward'] || '-')}</td>`;
            tableHtml += `<td>${this.escapeHtml(row['Saving Value (INR)'] || '-')}</td>`;
            tableHtml += `<td>${this.escapeHtml(row['Weight Saving (Kg)'] || '-')}</td>`;
            
            // Status
            const status = row['Status'] || 'TBD';
            let badgeClass = 'role-User';
            if(status.includes('AI')) badgeClass = 'badge-purple';
            if(status === 'Web Sourced') badgeClass = 'badge-info';
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
        resultsContainer.innerHTML = actionsHtml + tableHtml;

        this.attachExportListeners();
    }
    
    // Helper to keep code clean
    attachExportListeners() {
         const excelBtn = document.getElementById('downloadExcelBtn');
         if(excelBtn) excelBtn.addEventListener('click', () => this.downloadExcel());
         
         const pptBtn = document.getElementById('downloadPptBtn');
         if(pptBtn) pptBtn.addEventListener('click', () => this.downloadPpt());
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
            console.error('Excel generation error:', error);
            this.showNotification('Error generating Excel: ' + error.message, 'error');
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
                throw new Error(result.error || 'Failed to generate PPT file');
            }
        } catch (error) {
            console.error('PPT generation error:', error);
            this.showNotification('Error generating PPT: ' + error.message, 'error');
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
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            background: ${type === 'success' ? '#4caf50' : type === 'error' ? '#f44336' : '#2196f3'};
            color: white;
            border-radius: 4px;
            z-index: 10000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
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

            if (data.history && data.history.length > 0) {
                let html = '<div class="activity-list">';
                
                data.history.forEach(item => {
                    html += `
                        <div class="activity-item">
                            <div class="activity-icon">
                                <i class="ph ph-chat-circle"></i>
                            </div>
                            <div class="activity-content">
                                <div class="activity-text">${this.escapeHtml(item.message || 'No message')}</div>
                                <div class="activity-time">${this.escapeHtml(item.timestamp || 'Unknown time')}</div>
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

    // ========================================
    // Database Management
    // ========================================

    async handleDatabaseUpload(e) {
        e.preventDefault();

        const excelInput = document.getElementById('databaseExcelFile');
        const zipInput = document.getElementById('databaseZipFile');

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

        try {
            const response = await fetch('/upload_database', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.showFlash(data.message || 'Database uploaded successfully', 'success');
                excelInput.value = '';
                zipInput.value = '';
                document.getElementById('databaseExcelFileName').textContent = 'No file chosen';
                document.getElementById('databaseZipFileName').textContent = 'No file chosen';
                this.loadDatabaseStatus();
                this.loadDatabaseUploads();
            } else {
                this.showFlash(data.error || 'Upload failed', 'error');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showFlash('Network error during upload', 'error');
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

        body.innerHTML = `<tr><td colspan="5" class="text-muted">Loading...</td></tr>`;

        try {
            const res = await fetch('/database_uploads');
            const data = await res.json();
            if (!data.success) {
                body.innerHTML = `<tr><td colspan="5" class="text-muted">${this.escapeHtml(data.error || 'Failed to load uploads')}</td></tr>`;
                return;
            }

            if (!data.uploads || data.uploads.length === 0) {
                body.innerHTML = `<tr><td colspan="5" class="text-muted">No uploads yet.</td></tr>`;
                return;
            }

            let html = '';
            data.uploads.forEach(u => {
                html += `
                    <tr>
                        <td>${this.escapeHtml(String(u.created_at || ''))}</td>
                        <td>${this.escapeHtml(u.excel_filename || '')}</td>
                        <td>${this.escapeHtml(u.zip_filename || '')}</td>
                        <td>${this.escapeHtml(u.uploaded_by || '')}</td>
                        <td>${this.escapeHtml(u.status || '')}</td>
                    </tr>
                `;
            });
            body.innerHTML = html;
        } catch (err) {
            console.error('Database uploads error:', err);
            body.innerHTML = `<tr><td colspan="5" class="text-muted">Error loading uploads</td></tr>`;
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
                this.showFlash(`Password reset. New password: ${data.new_password}`, 'success');
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
                container.innerHTML = `<pre>${this.escapeHtml(data.logs)}</pre>`;
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
                            <div class="bar-row">
                                <span class="bar-label">${this.escapeHtml(d.dept || 'Unknown')}</span>
                                <div class="bar-track">
                                    <div class="bar-fill" style="width:${width}%"></div>
                                </div>
                                <span class="bar-value">${d.idea_count}</span>
                            </div>
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
                                <div class="bar-row">
                                    <span class="bar-label">${this.escapeHtml(d.load_date || '')}</span>
                                    <div class="bar-track">
                                        <div class="bar-fill" style="width:${w}%"></div>
                                    </div>
                                    <span class="bar-value">${d.rows}</span>
                                </div>
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
                            <div class="activity-item">
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
                            </div>
                        `;
                    });
                    topSavingsEl.innerHTML = html;
                }
            }
        } catch (err) {
            console.error('Analytics load error:', err);
            if (deptChartEl) deptChartEl.innerHTML = '<p class="empty-text">Error loading analytics</p>';
        }
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
            <i class="ph ph-${iconMap[category]}"></i>
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