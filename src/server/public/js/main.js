// public/js/main.js

// A simple map: tab ID -> partial file path
const partialsMap = {
    Welcome:            '/partials/welcome.html',
    DataManager:        '/partials/dataManager.html',
    PhoneticModeling:   '/partials/phoneticModeling.html',
    DataViewer:         '/partials/dataViewer.html'
  };
  
document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tablinks');
    const container = document.getElementById('tab-content-container');
    const allNotesDiv = document.getElementById('AllNotes');
    
  
    // Initially, load the "Welcome" partial into the container
    loadPartial('Welcome');
    // Ensure AllNotes is hidden by default (if not the default tab)
    if (allNotesDiv) allNotesDiv.style.display = 'none';
    
    import('./notes/main_notes.js')
    .then(({ initNotes }) => initNotes())
    .catch(err => console.error('Failed to load notes module:', err));
    

    // Add click listeners for each main tab button
    tabs.forEach(tab => {
      tab.addEventListener('click', async (event) => {
        // Remove "active" from all tabs
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
  
        const tabName = tab.getAttribute('data-tab');
  
        // Special handling for AllNotes: Show the AllNotes div, hide the container.
        if (tabName === 'AllNotes') {
          container.style.display = 'none';
          if (allNotesDiv) allNotesDiv.style.display = 'block';
        } else {
          // For other tabs, ensure AllNotes is hidden and container is visible
          if (allNotesDiv) allNotesDiv.style.display = 'none';
          container.style.display = 'block';
          await loadPartial(tabName);
        }
      });
    });
  
    // Optionally, set the first tab as active (if it isn't AllNotes)
    if (tabs.length > 0 && tabs[0].getAttribute('data-tab') !== 'AllNotes') {
      tabs[0].classList.add('active');
    }
  
    async function loadPartial(tabName) {
      const url = partialsMap[tabName];
      if (!url) return;
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const html = await response.text();
        container.innerHTML = html;
          // Find the injected .tabcontent and add .active:
        const injected = container.querySelector('.tabcontent');
        if (injected) {
            injected.classList.add('active');
        }

        initializeTab(tabName);
      } catch (err) {
        console.error(`Error loading partial for ${tabName}:`, err);
        container.innerHTML = `<p style="color:red;">Error loading content for ${tabName}.</p>`;
      }
      

    }
  
    function initializeTab(tabId) {
      switch (tabId) {
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
            .catch(error => console.error('Error loading Data Manager module:', error));
          break;
        default:
          console.info(`No specific initialization needed for tab: ${tabId}`);
      }
    }
    
  });


  