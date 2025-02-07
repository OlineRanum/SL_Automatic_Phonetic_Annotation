// video_data_viewer.js (or wherever you keep your “Video Data” logic)

export function initVideoDataViewer() {
    const videoDataViewer = document.getElementById('videoDataViewerContent');
    if (!videoDataViewer) {
      console.error('Video Data Viewer content element not found.');
      return;
    }
  
    // Elements for listing and playing video
    const videoFileSelect = document.getElementById('video-file-select');
    const videoPlayer = document.getElementById('video-player');
  
    // 1) Fetch the list of video files from the server
    fetch('/api/data/video')
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to fetch video list: HTTP ${response.status}`);
        }
        return response.json();
      })
      .then(videoFiles => {
        if (!videoFiles.length) {
          videoFileSelect.innerHTML = '<option value="">No videos available</option>';
          return;
        }
        // Populate the <select>
        videoFileSelect.innerHTML = '<option value="">--Select a Video--</option>';
        videoFiles.forEach(file => {
          const option = document.createElement('option');
          option.value = file;
          option.textContent = file;
          videoFileSelect.appendChild(option);
        });
      })
      .catch(error => {
        console.error('Error fetching videos:', error);
        videoFileSelect.innerHTML = '<option value="">Error loading videos</option>';
      });
  
    // 2) Handle user selection in the <select>
    videoFileSelect.addEventListener('change', () => {
      const selectedFile = videoFileSelect.value;
      if (!selectedFile) {
        // If user picks the empty option, clear the <video> player
        videoPlayer.src = '';
        return;
      }
  
      // Set the video player's source to the chosen file
      // Assuming your server serves them from "/data/video/<filename>"
      videoPlayer.src = `/data/video/${encodeURIComponent(selectedFile)}`;
      videoPlayer.load();  // In some browsers, this ensures the new source is fully loaded
      videoPlayer.play();  // Autoplay if you want
    });
  }
  