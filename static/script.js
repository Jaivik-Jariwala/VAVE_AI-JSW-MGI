document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('userInput');
    const imageInput = document.getElementById('imageInput');
    const preview = document.getElementById('preview');
    const chatBtn = document.getElementById('chat-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const resultDiv = document.getElementById('result');
    const notification = document.getElementById('notification');

    // Fetch stats on page load
    async function fetchStats() {
        try {
            const response = await fetch('/stats');
            const data = await response.json();
            if (data.error) {
                showNotification('Error fetching stats: ' + data.error, 'error');
                return;
            }
            document.getElementById('total-ideas').textContent = data.total_ideas || 'N/A';
            document.getElementById('total-savings').textContent = data.total_savings || 'N/A';
            document.getElementById('ok-count').textContent = data.status_distribution?.OK || 'N/A';
            document.getElementById('tbd-count').textContent = data.status_distribution?.TBD || 'N/A';
            document.getElementById('ng-count').textContent = data.status_distribution?.NG || 'N/A';
        } catch (error) {
            showNotification('Error fetching stats: ' + error.message, 'error');
        }
    }

    // Image preview
    imageInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else {
            preview.style.display = 'none';
        }
    });

    // Chat button handler
    chatBtn.addEventListener('click', async function() {
        const query = userInput.value.trim();
        const file = imageInput.files[0];
        if (!query && !file) {
            showNotification('Please enter a query or select an image', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('user_input', query);
        if (file) {
            formData.append('image', file);
        }

        try {
            resultDiv.innerHTML = '<p>Loading...</p>';
            const response = await fetch('/chat', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.error) {
                resultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                showNotification('Error: ' + data.error, 'error');
                return;
            }

            let html = `<h3>Response:</h3><div class="response-content">${data.response}</div>`;
            if (data.table_data && data.table_data.length > 0) {
                html += '<h3>Results:</h3>';
                html += '<table class="data-table"><thead><tr>';
                Object.keys(data.table_data[0]).forEach(key => {
                    html += `<th>${key}</th>`;
                });
                html += '</tr></thead><tbody>';
                data.table_data.forEach(row => {
                    html += '<tr>';
                    Object.values(row).forEach(value => {
                        html += `<td>${value || ''}</td>`;
                    });
                    html += '</tr>';
                });
                html += '</tbody></table>';
            }
            if (data.image_urls?.proposal?.length > 0) {
                html += '<h3>Proposal Images:</h3><div class="image-gallery">';
                data.image_urls.proposal.forEach(url => {
                    html += `<img src="${url}" alt="Proposal Image">`;
                });
                html += '</div>';
            }
            if (data.image_urls?.mg?.length > 0) {
                html += '<h3>MG Product Images:</h3><div class="image-gallery">';
                data.image_urls.mg.forEach(url => {
                    html += `<img src="${url}" alt="MG Product Image">`;
                });
                html += '</div>';
            }
            if (data.image_urls?.competitor?.length > 0) {
                html += '<h3>Competitor Product Images:</h3><div class="image-gallery">';
                data.image_urls.competitor.forEach(url => {
                    html += `<img src="${url}" alt="Competitor Product Image">`;
                });
                html += '</div>';
            }
            if (data.csv_url) {
                html += `<p><a href="${data.csv_url}" download class="download-link">Download CSV</a></p>`;
            }
            if (data.ppt_url) {
                html += `<p><a href="${data.ppt_url}" download class="download-link">Download PPT</a></p>`;
            }
            if (data.audio) {
                html += `<p><audio controls><source src="${data.audio}" type="audio/mpeg">Your browser does not support the audio element.</audio></p>`;
            }
            resultDiv.innerHTML = html;
            showNotification('Query processed successfully', 'success');
        } catch (error) {
            resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            showNotification('Error: ' + error.message, 'error');
        }
    });

    // Upload button handler
    uploadBtn.addEventListener('click', async function() {
        const file = imageInput.files[0];
        if (!file) {
            showNotification('Please select an image', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('image', file);

        try {
            resultDiv.innerHTML = '<p>Uploading...</p>';
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.message && !data.message.includes('Error')) {
                let html = `<h3>Image Analysis:</h3><div class="response-content">${data.message}</div>`;
                if (data.table_data && data.table_data.length > 0) {
                    html += '<h3>Results:</h3>';
                    html += '<table class="data-table"><thead><tr>';
                    Object.keys(data.table_data[0]).forEach(key => {
                        html += `<th>${key}</th>`;
                    });
                    html += '</tr></thead><tbody>';
                    data.table_data.forEach(row => {
                        html += '<tr>';
                        Object.values(row).forEach(value => {
                            html += `<td>${value || ''}</td>`;
                        });
                        html += '</tr>';
                    });
                    html += '</tbody></table>';
                }
                if (data.image_urls?.proposal?.length > 0) {
                    html += '<h3>Proposal Images:</h3><div class="image-gallery">';
                    data.image_urls.proposal.forEach(url => {
                        html += `<img src="${url}" alt="Proposal Image">`;
                    });
                    html += '</div>';
                }
                if (data.image_urls?.mg?.length > 0) {
                    html += '<h3>MG Product Images:</h3><div class="image-gallery">';
                    data.image_urls.mg.forEach(url => {
                        html += `<img src="${url}" alt="MG Product Image">`;
                    });
                    html += '</div>';
                }
                if (data.image_urls?.competitor?.length > 0) {
                    html += '<h3>Competitor Product Images:</h3><div class="image-gallery">';
                    data.image_urls.competitor.forEach(url => {
                        html += `<img src="${url}" alt="Competitor Product Image">`;
                    });
                    html += '</div>';
                }
                resultDiv.innerHTML = html;
                showNotification('Image processed successfully', 'success');
            } else {
                resultDiv.innerHTML = `<p class="error">Error: ${data.message}</p>`;
                showNotification('Error: ' + data.message, 'error');
            }
        } catch (error) {
            resultDiv.innerHTML = `<p class="error">Error uploading image: ${error.message}</p>`;
            showNotification('Error: ' + error.message, 'error');
        }
    });

    // Notification function
    function showNotification(message, type) {
        notification.textContent = message;
        notification.className = type;
        notification.style.display = 'block';
        setTimeout(() => {
            notification.style.display = 'none';
            notification.className = '';
        }, 3000);
    }

    // Load stats on page load
    fetchStats();
});