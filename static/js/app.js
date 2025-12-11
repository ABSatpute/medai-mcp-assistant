// Modern Medical App JavaScript
class MedAI {
    constructor() {
        console.log('🚀 Initializing MedAI...');
        this.socket = io();
        this.currentThreadId = null;
        this.currentImage = null;
        this.userLocation = null;
        this.map = null;
        this.init();
    }

    init() {
        console.log('📡 Setting up MedAI components...');
        this.setupSocketListeners();
        this.getUserLocation();
        this.loadThreads();
        this.newChat();
        console.log('✅ MedAI initialized successfully');
    }

    getUserLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    this.userLocation = {
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude
                    };
                    console.log('Location detected:', this.userLocation);
                },
                (error) => {
                    console.log('Location access denied:', error);
                }
            );
        }
    }
                    this.userLocation = {latitude: 18.558091, longitude: 73.793439};
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 300000
                }
            );
        } else {
            console.log('Geolocation not supported');
            this.userLocation = {latitude: 18.558091, longitude: 73.793439};
        }
    }

    setupSocketListeners() {
        console.log('🔌 Setting up socket listeners...');
        
        this.socket.on('connect', () => {
            console.log('✅ Socket connected to server');
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Socket disconnected from server');
        });
        
        this.socket.on('message_received', (data) => {
            console.log('📨 Message received:', data);
            this.addMessage(data.role, data.content, data.timestamp, data.image);
            this.hideTypingIndicator();
        });

        this.socket.on('message_stream', (data) => {
            this.updateStreamingMessage(data);
            if (data.is_complete) {
                this.hideTypingIndicator();
            }
        });

        this.socket.on('title_updated', (data) => {
            this.updateThreadTitle(data.thread_id, data.title);
            if (data.thread_id === this.currentThreadId) {
                document.getElementById('chatTitle').textContent = data.title;
            }
        });

        this.socket.on('show_map', (data) => {
            this.showMap(data.stores, data.user_location);
        });
        });

        this.socket.on('medicine_results', (data) => {
            this.showMedicineResults(data);
            this.hideTypingIndicator();
        });
    }

    async loadThreads() {
        try {
            const response = await fetch('/api/threads');
            const threads = await response.json();
            
            const threadsList = document.getElementById('threadsList');
            threadsList.innerHTML = '';
            
            threads.forEach(thread => {
                const threadElement = document.createElement('div');
                threadElement.className = 'thread-item';
                threadElement.innerHTML = `
                    <div class="thread-title">${thread.title}</div>
                    <div class="thread-actions">
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteThread('${thread.id}', event)">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                `;
                threadElement.addEventListener('click', (e) => {
                    if (!e.target.closest('.thread-actions')) {
                        this.loadThread(thread.id, e);
                    }
                });
                threadsList.appendChild(threadElement);
            });
        } catch (error) {
            console.error('Error loading threads:', error);
        }
    }

    async loadThread(threadId, event) {
        try {
            this.currentThreadId = threadId;
            
            // Update active thread
            document.querySelectorAll('.thread-item').forEach(item => {
                item.classList.remove('active');
            });
            event.target.closest('.thread-item').classList.add('active');
            
            // Load messages
            const response = await fetch(`/api/thread/${threadId}/messages`);
            const messages = await response.json();
            
            this.clearMessages();
            messages.forEach(msg => {
                this.addMessage(msg.role, msg.content);
            });

            // Update title
            const threadTitle = event.target.closest('.thread-item').querySelector('.thread-title').textContent;
            document.getElementById('chatTitle').textContent = threadTitle;

        } catch (error) {
            console.error('Error loading thread:', error);
        }
    }

    newChat() {
        this.currentThreadId = this.generateUUID();
        this.clearMessages();
        document.getElementById('chatTitle').textContent = 'New Chat';
        
        // Remove active class from all threads
        document.querySelectorAll('.thread-item').forEach(item => {
            item.classList.remove('active');
        });

        // Show welcome message
        this.showWelcomeMessage();
    }

    showWelcomeMessage() {
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <div class="text-center">
                    <i class="fas fa-stethoscope fa-3x text-primary mb-3"></i>
                    <h4>Welcome to MedAI</h4>
                    <p class="text-muted">Your intelligent medical assistant</p>
                    <div class="welcome-cards">
                        <div class="welcome-card" onclick="medAI.quickAction('medicine')">
                            <i class="fas fa-pills"></i>
                            <h6>Medicine Search</h6>
                            <p>Find medicines and check availability</p>
                        </div>
                        <div class="welcome-card" onclick="medAI.quickAction('stores')">
                            <i class="fas fa-map-marker-alt"></i>
                            <h6>Store Locator</h6>
                            <p>Find nearby medical stores</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    clearMessages() {
        document.getElementById('chatMessages').innerHTML = '';
    }

    addMessage(role, content, timestamp = null, image = null) {
        const messagesContainer = document.getElementById('chatMessages');
        
        // Hide welcome message when first message is added
        const welcomeMessage = document.getElementById('welcomeMessage');
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }
        
        // Remove welcome message if present
        const welcomeMessageElement = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessageElement) {
            welcomeMessageElement.remove();
        }

        const messageElement = document.createElement('div');
        messageElement.className = `message ${role} fade-in`;
        
        const avatar = role === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-robot"></i>';

        let imageHtml = '';
        if (image) {
            imageHtml = `<div class="message-image"><img src="${image}" alt="Uploaded image" style="max-width: 200px; border-radius: 8px; margin-bottom: 8px;"></div>`;
        }

        messageElement.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                ${imageHtml}
                <div class="message-text">${this.formatMessage(content)}</div>
            </div>
        `;

        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    updateStreamingMessage(data) {
        const messagesContainer = document.getElementById('chatMessages');
        
        // Remove welcome message if present
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        // Find or create streaming message
        let streamingMessage = messagesContainer.querySelector('.streaming-message');
        if (!streamingMessage) {
            streamingMessage = document.createElement('div');
            streamingMessage.className = 'message assistant streaming-message fade-in';
            streamingMessage.innerHTML = `
                <div class="message-avatar"><i class="fas fa-robot"></i></div>
                <div class="message-content">
                    <div class="message-text"></div>
                </div>
            `;
            messagesContainer.appendChild(streamingMessage);
        }

        // Update content
        const messageText = streamingMessage.querySelector('.message-text');
        messageText.innerHTML = this.formatMessage(data.content);
        
        if (data.is_complete) {
            streamingMessage.classList.remove('streaming-message');
        }

        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    formatMessage(content) {
        // Basic markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    showTypingIndicator() {
        const messagesContainer = document.getElementById('chatMessages');
        const typingElement = document.createElement('div');
        typingElement.className = 'typing-indicator';
        typingElement.id = 'typingIndicator';
        typingElement.innerHTML = `
            <div class="message assistant">
                <div class="message-avatar"><i class="fas fa-robot"></i></div>
                <div class="message-content">
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;
        messagesContainer.appendChild(typingElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    sendMessage() {
        console.log('📤 SendMessage called');
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        console.log('Message to send:', message);
        console.log('Current thread ID:', this.currentThreadId);
        console.log('Socket connected:', this.socket.connected);
        
        if (!message && !this.currentImage) {
            console.log('❌ No message or image to send');
            return;
        }

        // Show typing indicator immediately
        this.showTypingIndicator();

        // Send to server
        console.log('🚀 Emitting send_message event...');
        this.socket.emit('send_message', {
            thread_id: this.currentThreadId,
            message: message,
            image: this.currentImage,
            user_location: this.userLocation
        });

        console.log('✅ Message sent to server');

        // Clear input and image
        messageInput.value = '';
        this.removeImage();
    }

    handleKeyPress(event) {
        if (event.key === 'Enter') {
            this.sendMessage();
        }
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImage = e.target.result;
            this.showImagePreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    showImagePreview(imageSrc) {
        const previewContainer = document.getElementById('imagePreview');
        previewContainer.innerHTML = `
            <div class="image-preview-item">
                <img src="${imageSrc}" alt="Preview" class="preview-image">
                <button type="button" class="btn btn-sm btn-danger remove-image" onclick="removeImage()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        previewContainer.style.display = 'block';
    }

    removeImage() {
        this.currentImage = null;
        const previewContainer = document.getElementById('imagePreview');
        previewContainer.innerHTML = '';
        previewContainer.style.display = 'none';
        
        // Reset file input
        document.getElementById('imageInput').value = '';
    }

    updateThreadTitle(threadId, newTitle) {
        const threadItems = document.querySelectorAll('.thread-item');
        threadItems.forEach(item => {
            const titleElement = item.querySelector('.thread-title');
            if (titleElement && item.addEventListener) {
                // Find the thread by checking if it matches our current thread
                if (this.currentThreadId === threadId) {
                    titleElement.textContent = newTitle;
                }
            }
        });
    }

    async deleteThread(threadId, event) {
        event.stopPropagation();
        
        if (!confirm('Are you sure you want to delete this conversation?')) {
            return;
        }

        try {
            const response = await fetch(`/api/thread/${threadId}/delete`, {
                method: 'DELETE'
            });

            if (response.ok) {
                // Remove from UI
                event.target.closest('.thread-item').remove();
                
                // If this was the current thread, start a new chat
                if (this.currentThreadId === threadId) {
                    this.newChat();
                }
            }
        } catch (error) {
            console.error('Error deleting thread:', error);
        }
    }

    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    showStoreMap(stores, userLocation) {
        const mapContainer = document.getElementById('mapContainer');
        const mapOverlay = document.getElementById('mapOverlay');
        
        mapContainer.style.display = 'block';
        mapOverlay.style.display = 'block';

        // Initialize map if not exists
        if (!this.map) {
            this.map = L.map('map');
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(this.map);
        }

        // Clear existing markers and routes
        this.map.eachLayer((layer) => {
            if (layer instanceof L.Marker || layer instanceof L.Polyline) {
                this.map.removeLayer(layer);
            }
        });

        // Set map center to user location
        this.map.setView([userLocation.latitude, userLocation.longitude], 13);

        // Add user location marker (always visible)
        const userIcon = L.divIcon({
            html: '<div style="background: #2563eb; width: 16px; height: 16px; border-radius: 50%; border: 4px solid white; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4); position: relative;"><div style="position: absolute; top: -2px; left: -2px; width: 20px; height: 20px; border-radius: 50%; border: 2px solid #2563eb; animation: pulse 2s infinite;"></div></div>',
            iconSize: [24, 24],
            className: 'user-location-marker'
        });

        this.userMarker = L.marker([userLocation.latitude, userLocation.longitude], {icon: userIcon})
            .addTo(this.map)
            .bindPopup('<b>📍 Your Location</b>')
            .setZIndexOffset(1000); // Keep user marker on top

        // Add store markers with names
        stores.forEach((store, index) => {
            const isNearest = index === 0;
            
            // Create store name icon
            const storeName = store.store_name.length > 12 ? 
                store.store_name.substring(0, 12) + '...' : 
                store.store_name;
            
            const storeIcon = L.divIcon({
                html: `
                    <div style="
                        background: ${isNearest ? '#10b981' : '#ef4444'}; 
                        color: white; 
                        padding: 4px 8px; 
                        border-radius: 12px; 
                        font-size: 11px;
                        font-weight: 600;
                        white-space: nowrap;
                        border: 2px solid white; 
                        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                        text-align: center;
                        min-width: 60px;
                    ">
                        ${isNearest ? '🏆 ' : ''}${storeName}
                    </div>
                `,
                iconSize: [80, 24],
                iconAnchor: [40, 12],
                className: 'store-marker'
            });

            const marker = L.marker([store.latitude, store.longitude], {icon: storeIcon})
                .addTo(this.map);

            // Hover tooltip
            marker.bindTooltip(`
                <div style="min-width: 200px;">
                    <b>${isNearest ? '🏆 ' : ''}${store.store_name}</b><br>
                    📍 ${store.address}<br>
                    📞 ${store.phone_number}<br>
                    📏 ${store.distance.toFixed(2)} km away<br>
                    🕒 Open 9 AM - 10 PM
                </div>
            `, {permanent: false, direction: 'top'});

            // Click for directions
            marker.on('click', () => {
                this.showDirections(userLocation, store);
            });
        });
    }

    async showDirections(userLocation, store) {
        try {
            // Ensure user marker is visible
            if (this.userMarker) {
                this.userMarker.setZIndexOffset(1000);
            }

            // Get route from OSRM
            const osrmUrl = `https://router.project-osrm.org/route/v1/driving/${userLocation.longitude},${userLocation.latitude};${store.longitude},${store.latitude}?overview=full&geometries=geojson`;
            
            const response = await fetch(osrmUrl);
            const data = await response.json();

            if (data.routes && data.routes.length > 0) {
                const route = data.routes[0];
                const coordinates = route.geometry.coordinates.map(coord => [coord[1], coord[0]]);

                // Clear existing route
                this.map.eachLayer((layer) => {
                    if (layer instanceof L.Polyline) {
                        this.map.removeLayer(layer);
                    }
                });

                // Add route line
                L.polyline(coordinates, {
                    color: '#2563eb',
                    weight: 4,
                    opacity: 0.8
                }).addTo(this.map);

                // Show route info
                const distance = (route.distance / 1000).toFixed(2);
                const duration = Math.round(route.duration / 60);

                // Create route info popup
                const routeInfo = L.popup()
                    .setLatLng([store.latitude, store.longitude])
                    .setContent(`
                        <div style="text-align: center;">
                            <b>🚗 Route to ${store.store_name}</b><br>
                            📏 Distance: ${distance} km<br>
                            ⏱️ Time: ${duration} minutes
                        </div>
                    `)
                    .openOn(this.map);

            } else {
                alert('Unable to find route. Showing straight line distance.');
            }
        } catch (error) {
            console.error('Routing error:', error);
            alert('Unable to get directions. Please try again.');
        }
    }

    showMedicineResults(data) {
        const messagesContainer = document.getElementById('chatMessages');
        
        let resultsHtml = `
            <div class="medicine-results">
                <h3><i class="fas fa-pills"></i> Medicine Search Results</h3>
        `;

        // Show search summary
        if (data.searched_medicines && data.searched_medicines.length > 0) {
            resultsHtml += `<div class="search-summary">`;
            resultsHtml += `<p><strong>Searched for:</strong> ${data.searched_medicines.join(', ')}</p>`;
            
            if (data.found_medicines && data.found_medicines.length > 0) {
                resultsHtml += `<p class="found-medicines"><i class="fas fa-check-circle"></i> <strong>Found:</strong> ${data.found_medicines.join(', ')}</p>`;
            }
            
            if (data.not_found_medicines && data.not_found_medicines.length > 0) {
                resultsHtml += `<p class="not-found-medicines"><i class="fas fa-exclamation-triangle"></i> <strong>Not found:</strong> ${data.not_found_medicines.join(', ')}</p>`;
            }
            resultsHtml += `</div>`;
        }
        
        if (data.medicines && data.medicines.length > 0) {
            resultsHtml += `<p>Found ${data.medicines.length} medicine(s) available in nearby stores</p>`;

            data.medicines.forEach(medicine => {
                resultsHtml += `
                    <div class="medicine-card">
                        <div class="medicine-header">
                            <div>
                                <h4>${medicine.medicine_name}</h4>
                                ${medicine.searched_for ? `<small class="searched-for">Searched for: "${medicine.searched_for}"</small>` : ''}
                            </div>
                            <span class="price">₹${medicine.price}</span>
                        </div>
                        <p class="brand">${medicine.brand_name}</p>
                        <p class="description">${medicine.description}</p>
                        
                        <div class="availability-section">
                            <h5>Available at ${medicine.stores.length} store(s):</h5>
                            <div class="store-list">
                `;

                medicine.stores.forEach((store, index) => {
                    resultsHtml += `
                        <div class="store-item ${index === 0 ? 'nearest' : ''}">
                            <div class="store-info">
                                <strong>${index === 0 ? '🏆 ' : ''}${store.store_name}</strong>
                                <span class="distance">${store.distance.toFixed(2)} km away</span>
                            </div>
                            <div class="stock-info">
                                <span class="stock">Stock: ${store.stock_quantity}</span>
                                <button onclick="medAI.getDirectionsToStore(${store.latitude}, ${store.longitude}, '${store.store_name}')" class="directions-btn">
                                    <i class="fas fa-directions"></i> Directions
                                </button>
                            </div>
                        </div>
                    `;
                });

                resultsHtml += `
                            </div>
                        </div>
                    </div>
                `;
            });

            resultsHtml += '</div>';
            messagesContainer.innerHTML = resultsHtml;
        } else {
            messagesContainer.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-exclamation-circle"></i>
                    <h3>No medicines found</h3>
                    <p>None of the medicines you searched for are available in nearby stores.</p>
                    ${data.not_found_medicines && data.not_found_medicines.length > 0 ? 
                        `<p><strong>Not found:</strong> ${data.not_found_medicines.join(', ')}</p>` : ''}
                    <button onclick="medAI.openMedicineSearch()" class="retry-btn">Search Again</button>
                </div>
            `;
        }
    }

    getDirectionsToStore(lat, lng, storeName) {
        if (!this.userLocation) {
            alert('Location not available. Please enable location services.');
            return;
        }

        // Open map with specific store
        const storeData = [{
            store_name: storeName,
            latitude: lat,
            longitude: lng,
            distance: this.calculateDistance(this.userLocation.latitude, this.userLocation.longitude, lat, lng)
        }];

        this.showStoreMap(storeData, this.userLocation);
    }

    calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // Earth's radius in km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                  Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    fillExample(exampleText) {
        const input = document.getElementById('medicineSearchInput');
        const container = document.querySelector('.search-input-container');
        
        container.style.display = 'flex';
        input.value = exampleText;
        input.focus();
    }

    quickAction(type) {
        if (type === 'medicine') {
            this.openMedicineSearch();
        } else if (type === 'stores') {
            this.openStoreLocator();
        }
    }

    openMedicineSearch() {
        // Clear welcome message and show medicine search interface
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.innerHTML = `
            <div class="medicine-search-container">
                <div class="search-header">
                    <h3><i class="fas fa-pills"></i> Medicine Search</h3>
                    <p>Tell me the medicine name or upload a prescription photo</p>
                </div>
                
                <div class="search-options">
                    <div class="search-option" onclick="medAI.focusSearchInput()">
                        <i class="fas fa-keyboard"></i>
                        <h4>Type Medicine Name</h4>
                        <p>Enter the medicine you're looking for</p>
                    </div>
                    
                    <div class="search-option" onclick="medAI.uploadPrescription()">
                        <i class="fas fa-camera"></i>
                        <h4>Upload Prescription</h4>
                        <p>Take a photo of your prescription</p>
                    </div>
                </div>
                
                <div class="search-input-container" style="display: none;">
                    <input type="text" id="medicineSearchInput" placeholder="Enter medicine names (e.g., Paracetamol, Dolo 650, Aspirin)" />
                    <button onclick="medAI.searchMedicine()" class="search-btn">
                        <i class="fas fa-search"></i> Search
                    </button>
                </div>
                
                <div class="search-examples">
                    <p><strong>Examples:</strong></p>
                    <div class="example-tags">
                        <span class="example-tag" onclick="medAI.fillExample('Paracetamol')">Paracetamol</span>
                        <span class="example-tag" onclick="medAI.fillExample('Dolo 650, Crocin')">Dolo 650, Crocin</span>
                        <span class="example-tag" onclick="medAI.fillExample('Aspirin and Vitamin C')">Aspirin and Vitamin C</span>
                    </div>
                </div>
            </div>
        `;
    }

    focusSearchInput() {
        const container = document.querySelector('.search-input-container');
        const input = document.getElementById('medicineSearchInput');
        
        container.style.display = 'flex';
        input.focus();
        
        // Add enter key listener
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchMedicine();
            }
        });
    }

    uploadPrescription() {
        document.getElementById('imageInput').click();
        
        // Override the normal image handler for prescription mode
        const imageInput = document.getElementById('imageInput');
        imageInput.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    this.processPrescriptionImage(event.target.result);
                };
                reader.readAsDataURL(file);
            }
        };
    }

    async processPrescriptionImage(imageData) {
        this.showTypingIndicator();
        
        // Send prescription image for analysis
        this.socket.emit('analyze_prescription', {
            thread_id: this.currentThreadId,
            image: imageData,
            user_location: this.userLocation
        });
    }

    async searchMedicine() {
        const input = document.getElementById('medicineSearchInput');
        const medicineName = input.value.trim();
        
        if (!medicineName) {
            alert('Please enter a medicine name');
            return;
        }

        this.showTypingIndicator();
        
        // Send medicine search request
        this.socket.emit('search_medicine', {
            thread_id: this.currentThreadId,
            medicine_name: medicineName,
            user_location: this.userLocation
        });
    }

    async openStoreLocator() {
        // Ask for location permission
        if (!navigator.geolocation) {
            alert('Geolocation is not supported by this browser.');
            return;
        }

        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 300000
                });
            });

            const userLocation = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude
            };

            console.log('User location:', userLocation);

            // Fetch nearby stores
            const response = await fetch('/api/nearby-stores', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(userLocation)
            });

            const stores = await response.json();
            console.log('Fetched stores:', stores);

            if (stores.length === 0) {
                alert('No stores found in your area.');
                return;
            }

            this.showStoreMap(stores, userLocation);

        } catch (error) {
            console.error('Location error:', error);
            alert('Unable to get your location. Please enable location services.');
        }
    }
}

// Global functions for HTML event handlers
let medAI;

function sendMessage() {
    medAI.sendMessage();
}

function newChat() {
    medAI.newChat();
}

function handleKeyPress(event) {
    medAI.handleKeyPress(event);
}

function handleImageUpload(event) {
    medAI.handleImageUpload(event);
}

function removeImage() {
    medAI.removeImage();
}

function deleteThread(threadId, event) {
    medAI.deleteThread(threadId, event);
}

function toggleSidebar() {
    console.log('Toggle sidebar clicked'); // Debug log
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    if (!sidebar || !mainContent) {
        console.error('Sidebar or main content not found');
        return;
    }
    
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');
    console.log('Sidebar toggled, collapsed:', sidebar.classList.contains('collapsed'));
}

function closeMap() {
    document.getElementById('mapContainer').style.display = 'none';
    document.getElementById('mapOverlay').style.display = 'none';
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    medAI = new MedAI();
});

function sendQuickMessage(message) {
    console.log('Quick message clicked:', message);
    
    if (!medAI) {
        console.error('MedAI not initialized');
        alert('App not ready, please wait...');
        return;
    }
    
    const messageInput = document.getElementById('messageInput');
    if (!messageInput) {
        console.error('Message input not found');
        return;
    }
    
    messageInput.value = message;
    console.log('Calling medAI.sendMessage()');
    medAI.sendMessage();
}
