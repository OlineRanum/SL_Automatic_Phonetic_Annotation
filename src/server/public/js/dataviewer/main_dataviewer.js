// main_dataviewer.js
import { fetchAndDisplayNotes } from '../notes/main_notes.js';
import { showNotification } from '../notifications/main_notifications.js';
import { preloadBatch, startPlayback, stopPlayback, resetPlayInterval } from '../viewer_utils.js';

export function initDataViewer() {
    // -------- Data Viewer Elements --------
    const gifSelect      = document.getElementById('gif-select');
    const gifImage       = document.getElementById('gif-image');

    const prevBtn        = document.getElementById('prev-btn');
    const nextBtn        = document.getElementById('next-btn');
    const togglePlayBtn  = document.getElementById('toggle-play-btn');
    
    const frameInfo      = document.getElementById('frame-info');
    const frameRateInput = document.getElementById('frame-rate');
    const loadingIndicator= document.getElementById('loading');
    const progressBar    = document.getElementById('progress-bar');

    let currentGifName     = '';
    let currentFrameName   = 'frame_0000.png';
    let currentFrames      = [];
    let preloadedImages    = [];
    let currentFrameIndex  = 0;

    const isPlayingRef     = { value: false };
    const playIntervalRef  = { value: null };
    let frameRate          = 200;

    const preloadBatchSize = 1000;

    // ---------- (A) FETCH GIF LIST ----------
// main_dataviewer.js

function fetchGifs() {
    fetch('/api/gifs', { cache: 'no-store' })
        .then(response => response.json())
        .then(gifs => {
            if (gifs.length === 0) {
                gifSelect.innerHTML = '<option value="">No GIFs available</option>';
                return;
            }
            // Clear existing options
            gifSelect.innerHTML = '<option value="">--Select a GIF--</option>';
            gifs.forEach(gif => {
                const option = document.createElement('option');
                const baseName = gif.endsWith('.gif') ? gif.slice(0, -4) : gif; // Remove '.gif' if present
                option.value = baseName; // Exclude '.gif'
                option.textContent = gif; // Display with '.gif'
                gifSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error fetching GIFs:', error);
            gifSelect.innerHTML = '<option value="">Error loading GIFs</option>';
        });
}



        // ---------- (B) ON GIF SELECTION ----------
    gifSelect.addEventListener('change', () => {
        const selectedGif = gifSelect.value;
        if (selectedGif) {
            currentGifName = selectedGif; 
            stopPlaybackImmediate();  // Stop if playing
            showLoading(true);

            fetch(`/api/gifs/${encodeURIComponent(selectedGif)}/frames`, { cache: 'no-store' })
                .then(response => {
                    if (!response.ok) throw new Error('Frames not found.');
                    return response.json();
                })
                .then(frames => {
                    if (frames.length === 0) throw new Error('No frames available for this GIF.');
                    currentFrames = frames;
                    currentFrameIndex = 0;
                    preloadedImages = new Array(currentFrames.length).fill(null);

                    // Start preloading
                    preloadBatch(
                      currentFrames,
                      preloadedImages,
                      0,
                      preloadBatchSize,
                      (percent) => updateProgress(percent)
                    ).then(endIndex => {
                        // We loaded the first batch. Let's show the first frame
                        updateFrame();
                        updateControls();

                        // If there are more frames after that batch, keep loading them in the background
                        continuePreloading(endIndex);
                    });
                })
                .catch(error => {
                    console.error('Error fetching frames:', error);
                    loadingIndicator.innerHTML = 'Error loading frames.';
                });
        } else {
            // Reset if no GIF is selected
            resetViewer();
        }
    });

    // Preload subsequent batches in background
    function continuePreloading(startIndex) {
        if (startIndex >= currentFrames.length) {
            // All frames loaded
            showLoading(false);
            updateProgress(0);
            return;
        }
        preloadBatch(
            currentFrames,
            preloadedImages,
            startIndex,
            preloadBatchSize,
            (percent) => {
                // If it's the last batch, update the bar. Otherwise, we might keep it hidden 
                updateProgress(percent);
            }
        ).then(nextIndex => {
            if (nextIndex >= currentFrames.length) {
                showLoading(false);
                updateProgress(0);
            } else {
                continuePreloading(nextIndex);
            }
        }).catch(err => {
            console.error('Error preloading frames:', err);
            loadingIndicator.innerHTML = 'Error loading frames.';
            updateProgress(0);
        });
    }

    // ---------- (C) UPDATE FRAME ----------
    function updateFrame() {
        if (preloadedImages[currentFrameIndex]) {
            gifImage.src = preloadedImages[currentFrameIndex].src;
            const parts = gifImage.src.split('/');
            currentFrameName = parts[parts.length - 1];
        } else {
            gifImage.src = '';
        }
        frameInfo.textContent = `Frame: ${currentFrameIndex + 1} / ${preloadedImages.length}`;

        // Dispatch an event so the notes script can update
        const frameUpdateEvent = new CustomEvent('frameUpdated', {
            detail: { gifName: currentGifName, frameName: currentFrameName },
        });
        document.dispatchEvent(frameUpdateEvent);

        // Also fetch and display notes if frames are loaded
        if (currentGifName && currentFrameName) {
            fetchAndDisplayNotes(currentGifName, currentFrameName);
        }
    }

    // ---------- (D) PLAYBACK ----------
    function startPlaybackFn() {
        togglePlayBtn.textContent = 'Stop GIF';
        togglePlayBtn.classList.add('playing');
        togglePlayBtn.classList.remove('paused');
        playIntervalRef.value = startPlayback({
            isPlayingRef,
            frameRate,
            currentFrameIndexRef: { value: currentFrameIndex },
            preloadedImages,
            updateFrame: () => {
                currentFrameIndex = playIntervalRef.value
                  ? (playIntervalRef.value.currentFrameIndexRef?.value ?? currentFrameIndex)
                  : currentFrameIndex;
                updateFrame();
            },
            updateControls
        });
    }
    function stopPlaybackFn() {
        togglePlayBtn.textContent = 'Play GIF';
        togglePlayBtn.classList.add('paused');
        togglePlayBtn.classList.remove('playing');
        stopPlayback({ isPlayingRef, playIntervalRef });
    }
    function stopPlaybackImmediate() {
        isPlayingRef.value = false;
        if (playIntervalRef.value) {
            clearInterval(playIntervalRef.value);
            playIntervalRef.value = null;
        }
        togglePlayBtn.textContent = 'Play GIF';
    }

    // ---------- (E) CONTROLS ----------
    function updateControls() {
        if (preloadedImages.length > 0) {
            const isSingleFrame = (preloadedImages.length === 1);
            prevBtn.disabled        = isPlayingRef.value || currentFrameIndex === 0 || isSingleFrame;
            nextBtn.disabled        = isPlayingRef.value || currentFrameIndex === (preloadedImages.length - 1) || isSingleFrame;
            togglePlayBtn.disabled  = isSingleFrame;
        } else {
            prevBtn.disabled        = true;
            nextBtn.disabled        = true;
            togglePlayBtn.disabled  = true;
        }
    }

    // ---------- (F) PROGRESS BAR, LOADING UI ----------
    function showLoading(show) {
        loadingIndicator.style.display = show ? 'block' : 'none';
        if (show) {
            loadingIndicator.innerHTML = `
              <span>Loading frames...</span>
              <div id="progress-container"><div id="progress-bar"></div></div>
            `;
        }
    }
    function updateProgress(percent) {
        progressBar.style.width = `${percent}%`;
    }

    // ---------- (G) RESET & CLEAR ----------
    function resetViewer() {
        stopPlaybackImmediate();
        currentGifName    = '';
        currentFrameName  = '';
        currentFrames     = [];
        preloadedImages   = [];
        currentFrameIndex = 0;
        gifImage.src      = '';
        frameInfo.textContent = 'Frame: 0';
        updateControls();
        togglePlayBtn.disabled = true;
        showLoading(false);
        progressBar.style.width = '0%';
        clearNotesSection();
    }
    function clearNotesSection() {
        const notesList = document.getElementById('notes-list');
        if (!notesList) return;
        notesList.innerHTML = '<p>No GIF selected.</p>';
    }

    // ---------- (H) BUTTON LISTENERS ----------
    prevBtn.addEventListener('click', () => {
        if (currentFrameIndex > 0) {
            currentFrameIndex--;
            updateFrame();
            updateControls();
            if (isPlayingRef.value) {
                resetPlayInterval({
                    isPlayingRef,
                    playIntervalRef,
                    frameRate,
                    currentFrameIndexRef: { value: currentFrameIndex },
                    preloadedImages,
                    updateFrame: () => {
                        currentFrameIndex = playIntervalRef.value
                          ? playIntervalRef.value.currentFrameIndexRef?.value
                          : currentFrameIndex;
                        updateFrame();
                    },
                    updateControls
                });
            }
        }
    });
    nextBtn.addEventListener('click', () => {
        if (currentFrameIndex < preloadedImages.length - 1) {
            currentFrameIndex++;
            updateFrame();
            updateControls();
            if (isPlayingRef.value) {
                resetPlayInterval({
                    isPlayingRef,
                    playIntervalRef,
                    frameRate,
                    currentFrameIndexRef: { value: currentFrameIndex },
                    preloadedImages,
                    updateFrame: () => {
                        currentFrameIndex = playIntervalRef.value
                          ? playIntervalRef.value.currentFrameIndexRef?.value
                          : currentFrameIndex;
                        updateFrame();
                    },
                    updateControls
                });
            }
        }
    });
    togglePlayBtn.addEventListener('click', () => {
        if (!preloadedImages.length) return;
        if (isPlayingRef.value) {
            stopPlaybackFn();
            showNotification('Playback stopped.', false);
        } else {
            startPlaybackFn();
            showNotification('Playback started.', false);
        }
        updateControls();
    });

    // ---------- (I) FRAME RATE ----------
    frameRateInput.addEventListener('change', () => {
        const newRate = parseInt(frameRateInput.value, 10);
        if (!isNaN(newRate) && newRate >= 10 && newRate <= 1000) {
            frameRate = newRate;
            // If playing, restart interval
            if (isPlayingRef.value) {
                stopPlaybackImmediate();
                startPlaybackFn();
            }
        } else {
            frameRateInput.value = frameRate;
            showNotification('Invalid frame rate. Reset to previous value.', true);
        }
    });

    // ---------- (J) KEYBOARD NAVIGATION ----------
    document.addEventListener('keydown', (event) => {
        const tag = event.target.tagName.toLowerCase();
        if (tag === 'input' || tag === 'textarea') return;

        const dataViewerActive = document.getElementById('DataViewer').classList.contains('active');
        if (!dataViewerActive) return;

        switch (event.key) {
            case 'ArrowLeft':
                event.preventDefault();
                if (currentFrameIndex > 0) {
                    currentFrameIndex--;
                    updateFrame();
                    updateControls();
                    if (isPlayingRef.value) {
                        resetPlayInterval({
                            isPlayingRef,
                            playIntervalRef,
                            frameRate,
                            currentFrameIndexRef: { value: currentFrameIndex },
                            preloadedImages,
                            updateFrame: () => {
                                currentFrameIndex = playIntervalRef.value
                                  ? playIntervalRef.value.currentFrameIndexRef?.value
                                  : currentFrameIndex;
                                updateFrame();
                            },
                            updateControls
                        });
                    }
                }
                break;
            case 'ArrowRight':
                event.preventDefault();
                if (currentFrameIndex < preloadedImages.length - 1) {
                    currentFrameIndex++;
                    updateFrame();
                    updateControls();
                    if (isPlayingRef.value) {
                        resetPlayInterval({
                            isPlayingRef,
                            playIntervalRef,
                            frameRate,
                            currentFrameIndexRef: { value: currentFrameIndex },
                            preloadedImages,
                            updateFrame: () => {
                                currentFrameIndex = playIntervalRef.value
                                  ? playIntervalRef.value.currentFrameIndexRef?.value
                                  : currentFrameIndex;
                                updateFrame();
                            },
                            updateControls
                        });
                    }
                }
                break;
            case ' ':
                event.preventDefault();
                // Toggle play/pause
                if (isPlayingRef.value) {
                    stopPlaybackFn();
                    showNotification('Playback stopped.', false);
                } else {
                    if (preloadedImages.length > 0) {
                        startPlaybackFn();
                        showNotification('Playback started.', false);
                    }
                }
                updateControls();
                break;
            default:
                break;
        }
    });

    // ---------- (K) INIT -----------
    fetchGifs();
}
