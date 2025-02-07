import { fetchMoCapGifs } from './mocap_data_viewer.js';

export function initDataManager() {
    const mocapContainer = document.getElementById('DataManager');
    if (!mocapContainer) {
        console.error('DataManager container not found.');
        return;
    }

    const videoDataViewerTab = document.getElementById('videoDataViewerTab');
    const mocapDataViewerTab = document.getElementById('mocapDataViewerTab');
    const mocapDataUploaderTab = document.getElementById('dataUploaderTab');
    const videoDataViewerContent = document.getElementById('videoDataViewerContent');
    const mocapDataViewerContent = document.getElementById('mocapDataViewerContent');
    const dataUploaderContent = document.getElementById('dataUploaderContent');

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

    if (videoDataViewerTab && videoDataViewerContent) {
        videoDataViewerTab.addEventListener('click', () => {
            switchTab(videoDataViewerTab, videoDataViewerContent);
            if (!modulesLoaded.reference) {
                import('./video_data_viewer.js')
                    .then(({ initVideoDataViewer }) => {
                        initVideoDataViewer();
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

    if (mocapDataUploaderTab && dataUploaderContent) {
        mocapDataUploaderTab.addEventListener('click', () => {
            switchTab(mocapDataUploaderTab, dataUploaderContent);
            if (!modulesLoaded.uploader) {
                import('./data_uploader.js')
                    .then(({ initDataUploader }) => {
                        initDataUploader();
                        modulesLoaded.uploader = true;
                    })
                    .catch(err => console.error('Error loading Data Uploader:', err));
            }
        });
    }

    const defaultTab = mocapDataUploaderTab || referencePoseTab || mocapDataViewerTab;
    if (defaultTab) defaultTab.click();
}
