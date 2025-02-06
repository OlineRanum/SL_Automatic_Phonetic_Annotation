document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tablinks');
    const contents = document.querySelectorAll('.tabcontent');

    
    import('./notes/main_notes.js')
        .then(({ initNotes }) => initNotes())
        .catch(err => console.error('Failed to load notes:', err));

    // Tab navigation
    tabs.forEach(tab => {
        tab.addEventListener('click', (event) => {
            const targetTab = event.currentTarget.getAttribute('data-tab');

            // Deactivate all tabs
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.style.display = 'none');

            // Activate selected tab
            tab.classList.add('active');
            const content = document.getElementById(targetTab);
            if (content) {
                content.style.display = 'block';
                initializeTab(targetTab);
            }
        });
    });

    // Activate the first tab by default
    if (tabs.length > 0) {
        tabs[0].classList.add('active');
        const firstTab = tabs[0].getAttribute('data-tab');
        document.getElementById(firstTab).style.display = 'block';
        initializeTab(firstTab);
    }
});

// Initialize specific tabs dynamically
function initializeTab(tabId) {
    switch (tabId) {
        case 'AllNotes':
            break;
        case 'DataViewer':
            import('./dataviewer/main_dataviewer.js')
                .then(({ initDataViewer }) => initDataViewer())
                .catch(error => console.error('Error loading Data Viewer module:', error));
            break;
        case 'HandshapeViewer':
            import('./modules/handshape.js')
                .then(({ initHandshapeViewer }) => initHandshapeViewer())
                .catch(error => console.error('Error loading Handshape Viewer module:', error));
            break;
        case 'MoCap':
            import('./mocap/main_mocap.js')
                .then(({ initMoCap }) => initMoCap())
                .catch(error => console.error('Error loading MoCap module:', error));
            break;
        default:
            console.warn(`No initialization script found for tab: ${tabId}`);
    }
}
