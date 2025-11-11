class ChatbotApp {
    constructor() {
        this.currentSection = "chat";
        this.chatHistory = [];
        this.bindEvents();
        this.switchSection("chat");
        this.addMessageToChat(
            "Hello! I'm your AI assistant for cost-reduction analysis. Ask me anything.",
            "ai",
            this.getCurrentTime()
        );
    }

    bindEvents() {
        document.querySelectorAll(".nav-item").forEach((item) =>
            item.addEventListener("click", (e) => {
                e.preventDefault();
                this.switchSection(item.dataset.section);
            })
        );

        document
            .getElementById("sendBtn")
            .addEventListener("click", () => this.sendMessage());

        document
            .getElementById("chatInput")
            .addEventListener("keypress", (e) => {
                if (e.key === "Enter") this.sendMessage();
            });

        document
            .getElementById("sidebarToggle")
            .addEventListener("click", () => this.toggleSidebar());
    }

    async sendMessage() {
        const input = document.getElementById("chatInput");
        const msg = input.value.trim();
        if (!msg) return;

        this.addMessageToChat(msg, "user", this.getCurrentTime());
        input.value = "";
        this.showLoading();

        try {
            const formData = new FormData();
            formData.append('message', msg);

            const imageFile = document.getElementById('imageInput')?.files[0];
            if (imageFile) {
                formData.append('image', imageFile);
            }

            const res = await fetch("/chat", {
                method: "POST",
                body: formData,
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);

            const json = await res.json();
            console.log("Received response:", json); // Debug log

            if (!json.success) {
                throw new Error(json.error || "Unknown error occurred");
            }

            // Display brief confirmation in chat
            this.addMessageToChat(
                "✅ Analysis complete! Check the Results panel for detailed information.",
                "ai",
                this.getCurrentTime()
            );

            // Display results in the results panel
            console.log("Rendering table_data:", json.table_data); // Debug log
            this.displayResults(json.table_data || []);

            // Update download links
            this.updateDownloadLinks(json.csv_url, json.ppt_url);

            // Handle any warnings
            if (json.warning) {
                this.addMessageToChat(
                    `⚠️ Warning: ${json.warning}`,
                    "ai",
                    this.getCurrentTime()
                );
            }

        } catch (err) {
            console.error("Chat error:", err);
            this.addMessageToChat(
                `⚠️ Error: ${err.message}`,
                "ai",
                this.getCurrentTime()
            );
            // Ensure results panel shows error state
            this.displayResults([]);
        } finally {
            this.hideLoading();
        }
    }

    displayResults(rows) {
        const box = document.getElementById("resultsContent");
        if (!box) {
            console.error("Results content element not found");
            return;
        }

        console.log("Displaying results with rows:", rows); // Debug log

        if (!rows || rows.length === 0) {
            box.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📊</div>
                    <p>No results to display. Try a different query.</p>
                </div>
            `;
            return;
        }

        // Create table HTML
        const headers = [
            'Idea Id', 'Cost Reduction Idea', 'Proposal Image', 'MGI Gross Saving',
            'Way Forward', 'Estimated Cost Savings', 'MGI Carline',
            'Saving Value (INR)', 'Weight Saving (Kg)', 'Status', 'Dept', 'KD/LC'
        ];

        let tableHTML = `
            <div class="table-container">
                <table class="results-table">
                    <thead>
                        <tr>
                            ${headers.map(h => `<th>${h}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
        `;

        rows.forEach(row => {
            tableHTML += '<tr>';
            headers.forEach(header => {
                if (header === 'Proposal Image') {
                    const imgPath = row['Proposal Image Path'];
                    if (imgPath && imgPath !== 'N/A') {
                        tableHTML += `<td><img src="${imgPath}" alt="Proposal Image" style="max-width: 80px; max-height: 80px; object-fit: contain;"></td>`;
                    } else {
                        tableHTML += '<td>No Image</td>';
                    }
                } else {
                    const value = row[header] || 'N/A';
                    tableHTML += `<td>${value}</td>`;
                }
            });
            tableHTML += '</tr>';
        });

        tableHTML += `
                    </tbody>
                </table>
            </div>
        `;

        box.innerHTML = tableHTML;
        console.log("Table rendered successfully"); // Debug log
    }

    updateDownloadLinks(csvUrl, pptUrl) {
        const csvBtn = document.getElementById('downloadCsvBtn');
        const pptBtn = document.getElementById('downloadPptBtn');
        if (csvBtn && csvUrl) {
            csvBtn.href = csvUrl;
            csvBtn.download = 'cost_reduction_results.csv';
            csvBtn.classList.remove('hidden');
        }
        if (pptBtn && pptUrl) {
            pptBtn.href = pptUrl;
            pptBtn.download = 'cost_reduction_presentation.pptx';
            pptBtn.classList.remove('hidden');
        }
    }



    addMessageToChat(text, type, time) {
        const wrap = document.getElementById("chatMessages");
        const el = document.createElement("div");
        el.className = `message ${type}-message fade-in`;

        const avatar = type === 'ai' ? '🤖' : '👤';
        el.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <p>${text}</p>
                <span class="message-time">${time}</span>
            </div>
        `;

        wrap.appendChild(el);
        wrap.scrollTop = wrap.scrollHeight;
    }

    switchSection(section) {
        document.querySelectorAll(".content-section").forEach(s =>
            s.classList.remove("active")
        );

        document.querySelectorAll(".nav-item").forEach(item =>
            item.classList.remove("active")
        );

        document.getElementById(`${section}Section`).classList.add("active");
        document.querySelector(`[data-section="${section}"]`).classList.add("active");

        this.currentSection = section;
    }

    toggleSidebar() {
        const sidebar = document.querySelector(".sidebar");
        sidebar.classList.toggle("collapsed");
    }

    showLoading() {
        const loading = document.getElementById("loadingIndicator");
        if (loading) {
            loading.style.display = "flex";
        }
    }

    hideLoading() {
        const loading = document.getElementById("loadingIndicator");
        if (loading) {
            loading.style.display = "none";
        }
    }

    getCurrentTime() {
        return new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    new ChatbotApp();
});