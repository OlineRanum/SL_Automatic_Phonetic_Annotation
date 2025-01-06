// viewer_utils.js
// A small helper module to keep shared code DRY (Don't Repeat Yourself).

/**
 * Start GIF/Frame playback on a timer.
 * 
 * @param {Object} options 
 * @param {Object} options.isPlayingRef - An object containing a boolean `value` property. Will be set to true.
 * @param {number} options.frameRate - Interval in ms
 * @param {Object} options.currentFrameIndexRef - An object containing the current frame index in a `value` property. Will be incremented automatically.
 * @param {HTMLImageElement[]} options.preloadedImages
 * @param {Function} options.updateFrame - A callback to re-render the current frame in the UI
 * @param {Function} options.updateControls - A callback to update your UI controls (prev, next, etc.)
 * @returns {number} A setInterval() timer ID. Use clearInterval(...) to stop.
 */

export function preloadBatch(frames, preloadedImages, startIndex, batchSize, onProgress) {
    const endIndex   = Math.min(startIndex + batchSize, frames.length);
    const promises   = [];
    let loadedCount  = 0;
    const totalToLoad= endIndex - startIndex;
    
    for (let i = startIndex; i < endIndex; i++) {
      // Skip if already loaded
      if (preloadedImages[i]) {
        loadedCount++;
        onProgress((loadedCount / totalToLoad) * 100);
        continue;
      }
  
      const img = new Image();
      img.src = frames[i];
  
      const p = new Promise(resolve => {
        img.onload = () => {
          preloadedImages[i] = img;
          loadedCount++;
          onProgress((loadedCount / totalToLoad) * 100);
          resolve();
        };
        img.onerror = () => {
          // Even if it fails, continue
          loadedCount++;
          onProgress((loadedCount / totalToLoad) * 100);
          resolve();
        };
      });
      promises.push(p);
    }
  
    return Promise.all(promises).then(() => {
      // Return the endIndex so a caller can decide if more preloads are needed.
      return endIndex;
    });
  }
  
  /**
   * Start GIF/Frame playback on a timer.
   * 
   * @param {Object} options 
   * @param {boolean} options.isPlaying - Will be set to true
   * @param {number} options.frameRate - Interval in ms
   * @param {number} options.currentFrameIndex - Will be incremented automatically
   * @param {HTMLImageElement[]} options.preloadedImages
   * @param {Function} options.updateFrame - A callback to re-render the current frame in the UI
   * @param {Function} options.updateControls - A callback to update your UI controls (prev, next, etc.)
   * @returns {number} A setInterval() timer ID. Use clearInterval(...) to stop.
   */
  export function startPlayback({
    isPlayingRef,
    frameRate,
    currentFrameIndexRef,
    preloadedImages,
    updateFrame,
    updateControls
  }) {
    isPlayingRef.value = true;  // We store booleans in a small ref object so we can mutate them
    const intervalId = setInterval(() => {
      currentFrameIndexRef.value++;
      if (currentFrameIndexRef.value >= preloadedImages.length) {
        currentFrameIndexRef.value = 0;
      }
      updateFrame();
      updateControls();
    }, frameRate);
    return intervalId;
  }
  
  /**
   * Stop playback timer.
   * 
   * @param {Object} options
   * @param {boolean} options.isPlaying - Will be set to false
   * @param {number} options.playInterval - The current setInterval() ID
   */
  export function stopPlayback({ isPlayingRef, playIntervalRef }) {
    isPlayingRef.value = false;
    if (playIntervalRef.value) {
      clearInterval(playIntervalRef.value);
      playIntervalRef.value = null;
    }
  }
  
  /**
   * Reset the current playback interval if playing, to keep the user in sync
   */
  export function resetPlayInterval({
    isPlayingRef,
    playIntervalRef,
    frameRate,
    currentFrameIndexRef,
    preloadedImages,
    updateFrame,
    updateControls
  }) {
    if (!isPlayingRef.value) return;
    clearInterval(playIntervalRef.value);
    playIntervalRef.value = startPlayback({
      isPlayingRef,
      frameRate,
      currentFrameIndexRef,
      preloadedImages,
      updateFrame,
      updateControls
    });
  }
  