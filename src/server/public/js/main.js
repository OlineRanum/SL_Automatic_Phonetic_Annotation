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
            case 'PhoneticModeling':
                import('./phoneticmodelling/main_phonetics.js')
                  .then(({ initPhoneticModeling }) => initPhoneticModeling())
                  .catch(error => console.error('Error loading Phonetic Modeling module:', error));
                break;
        case 'DataManager':
            import('./datamanager/main_manager.js')
                .then(({ initDataManager }) => initDataManager())
                .catch(error => console.error('Error loading data manager module:', error));
            break;
        default:
            console.warn(`No initialization script found for tab: ${tabId}`);
    }
}
