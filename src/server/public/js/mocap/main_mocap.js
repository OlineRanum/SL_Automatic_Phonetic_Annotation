export function initMoCap() {
    const mocapContainer = document.getElementById('MoCap');

    if (!mocapContainer) {
        console.error('MoCap container not found.');
        return;
    }

    // Get tab elements
    const referencePoseTab = document.getElementById('referencePoseTab');
    const mocapDataViewerTab = document.getElementById('mocapDataViewerTab'); // Assuming there's a tab button for Data Viewer
    const referencePoseContent = document.getElementById('referencePoseContent'); // Assuming content sections have separate IDs
    const mocapDataViewerContent = document.getElementById('mocapDataViewerContent');

    // Load the Data Viewer tab by default
    import('./mocap_data_viewer.js')
        .then(({ initMoCapDataViewer }) => {
            initMoCapDataViewer();
        })
        .catch(err => console.error('Error loading default MoCap Data Viewer:', err));

    // Tab switching function
    function switchTab(activeTab, activeContent) {
        // Remove 'active' class from all tab buttons
        document.querySelectorAll('.mocap-tab-button').forEach(tab => tab.classList.remove('active'));
        
        // Remove 'active' class and hide all tab contents
        document.querySelectorAll('.mocap-tab-content').forEach(content => {
            content.classList.remove('active'); // Remove 'active' class
            content.style.display = 'none';      // Hide content
        });
    
        // Add 'active' class to the clicked tab button
        activeTab.classList.add('active');
        
        // Add 'active' class and display the corresponding tab content
        activeContent.classList.add('active');  // Add 'active' class to content
        activeContent.style.display = 'block';  // Show content
    }

    // Attach event listeners to tab buttons
    referencePoseTab.addEventListener('click', () => {
        switchTab(referencePoseTab, referencePoseContent);
        // Optionally, load specific modules or perform actions for this tab
        import('./reference_pose.js')
            .then(({ initReferencePose }) => {
                initReferencePose();
            })
            .catch(err => console.error('Error loading Reference Pose module:', err));
    });

    mocapDataViewerTab.addEventListener('click', () => {
        switchTab(mocapDataViewerTab, mocapDataViewerContent);
        // Optionally, reload or refresh the Data Viewer if needed
        import('./mocap_data_viewer.js')
            .then(({ initMoCapDataViewer }) => {
                initMoCapDataViewer();
            })
            .catch(err => console.error('Error loading MoCap Data Viewer:', err));
    });

    // Optionally, set the default active tab if not already set
    if (mocapDataViewerTab && mocapDataViewerContent) {
        switchTab(mocapDataViewerTab, mocapDataViewerContent);
    }
}
