// phonetic_modeling_manager.js

export function initPhoneticModeling() {
    const phoneticContainer    = document.getElementById('PhoneticModeling');
    if (!phoneticContainer) {
      console.error('PhoneticModeling container not found.');
      return;
    }
  
    // Sub-tab buttons
    const handshapesTab   = document.getElementById('handshapesTab');
    const locationTab     = document.getElementById('locationTab');
    const orientationTab  = document.getElementById('orientationTab');
  
    // Sub-tab contents
    const handshapesContent  = document.getElementById('handshapesContent');
    const locationContent    = document.getElementById('locationContent');
    const orientationContent = document.getElementById('orientationContent');
  
    // Keep track of which module(s) have been loaded
    // (handshapes, location, orientation). 
    // If you plan to lazy-load or code-split, you can
    // do import(...) the first time the user clicks.
    let modulesLoaded = {
      handshapes:   false,
      location:     false,
      orientation:  false,
    };
  
    // Switch sub-tab function
    function switchSubTab(activeTab, activeContent) {
      // Remove active from all sub-tab buttons
      document.querySelectorAll('.phonetic-tab-button').forEach(tabBtn => {
        tabBtn.classList.remove('active');
      });
  
      // Hide all sub-tab contents
      document.querySelectorAll('.phonetic-tab-content').forEach(tabContent => {
        tabContent.style.display = 'none';
      });
  
      // Activate the clicked tab & show corresponding content
      activeTab.classList.add('active');
      activeContent.style.display = 'block';
    }
  
    // ----- Handshapes Tab -----
    if (handshapesTab && handshapesContent) {
      handshapesTab.addEventListener('click', () => {
        switchSubTab(handshapesTab, handshapesContent);
  
        if (!modulesLoaded.handshapes) {
          // Example: lazy-load a "handshapes.js" module
          import('./handshapes.js')
            .then(({ initHandshapes }) => {
              initHandshapes();          // run the init
              modulesLoaded.handshapes = true;
            })
            .catch(err => console.error('Error loading handshapes module:', err));
        }
      });
    }
  
    // ----- Location Tab -----
    if (locationTab && locationContent) {
      locationTab.addEventListener('click', () => {
        switchSubTab(locationTab, locationContent);
  
        if (!modulesLoaded.location) {
          // Example: lazy-load a "location.js" module
          import('./location.js')
            .then(({ initLocation }) => {
              initLocation();
              modulesLoaded.location = true;
            })
            .catch(err => console.error('Error loading location module:', err));
        }
      });
    }
  
    // ----- Orientation Tab -----
    if (orientationTab && orientationContent) {
      orientationTab.addEventListener('click', () => {
        switchSubTab(orientationTab, orientationContent);
  
        if (!modulesLoaded.orientation) {
          // Example: lazy-load an "orientation.js" module
          import('./orientation.js')
            .then(({ initOrientation }) => {
              initOrientation();
              modulesLoaded.orientation = true;
            })
            .catch(err => console.error('Error loading orientation module:', err));
        }
      });
    }
  
    // Optionally auto-click the first sub-tab on load
    // If you want “Handshapes” by default:
    const defaultTab = handshapesTab || locationTab || orientationTab;
    if (defaultTab) defaultTab.click();
  }
  