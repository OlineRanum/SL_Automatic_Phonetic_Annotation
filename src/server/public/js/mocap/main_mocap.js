import { fetchMoCapGifs } from './mocap_data_viewer.js';

export function initMoCap() {
    const mocapContainer = document.getElementById('MoCap');
    if (!mocapContainer) {
        console.error('MoCap container not found.');
        return;
    }

    const referencePoseTab = document.getElementById('referencePoseTab');
    const mocapDataViewerTab = document.getElementById('mocapDataViewerTab');
    const mocapDataUploaderTab = document.getElementById('mocapDataUploaderTab');
    const referencePoseContent = document.getElementById('referencePoseContent');
    const mocapDataViewerContent = document.getElementById('mocapDataViewerContent');
    const mocapDataUploaderContent = document.getElementById('mocapDataUploaderContent');

    let modulesLoaded = {
        viewer: false,
        uploader: false,
        reference: false,
    };

    function switchTab(activeTab, activeContent) {
        document.querySelectorAll('.mocap-tab-button').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.mocap-tab-content').forEach(content => content.style.display = 'none');

        activeTab.classList.add('active');
        activeContent.style.display = 'block';
    }

    if (referencePoseTab && referencePoseContent) {
        referencePoseTab.addEventListener('click', () => {
            switchTab(referencePoseTab, referencePoseContent);
            if (!modulesLoaded.reference) {
                import('./reference_pose_selector.js')
                    .then(({ initReferencePose }) => {
                        initReferencePose();
                        modulesLoaded.reference = true;
                    })
                    .catch(err => console.error('Error loading Reference Pose module:', err));
            }
        });
    }

    if (mocapDataViewerTab && mocapDataViewerContent) {
        mocapDataViewerTab.addEventListener('click', () => {
            switchTab(mocapDataViewerTab, mocapDataViewerContent);
            fetchMoCapGifs();
            if (!modulesLoaded.viewer) {
                import('./mocap_data_viewer.js')
                    .then(({ initMoCapDataViewer }) => {
                        initMoCapDataViewer();
                        modulesLoaded.viewer = true;
                    })
                    .catch(err => console.error('Error loading MoCap Data Viewer:', err));
            }
        });
    }

    if (mocapDataUploaderTab && mocapDataUploaderContent) {
        mocapDataUploaderTab.addEventListener('click', () => {
            switchTab(mocapDataUploaderTab, mocapDataUploaderContent);
            if (!modulesLoaded.uploader) {
                import('./mocap_data_uploader.js')
                    .then(({ initMoCapDataUploader }) => {
                        initMoCapDataUploader();
                        modulesLoaded.uploader = true;
                    })
                    .catch(err => console.error('Error loading MoCap Data Uploader:', err));
            }
        });
    }

    const defaultTab = mocapDataViewerTab || referencePoseTab || mocapDataUploaderTab;
    if (defaultTab) defaultTab.click();
}
