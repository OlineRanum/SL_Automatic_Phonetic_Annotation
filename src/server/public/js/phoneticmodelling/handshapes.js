import { initReferencePoseHSViewer, initClusterSubTab } from './handshape_modules/handshapes_modules.js';

export function initHandshapes() {
  // 1. Initialize your sub-subtab menu
  const menuItems = document.querySelectorAll('#handshapes-subtab-menu li');
  const contents  = document.querySelectorAll('.sub-sub-tab-content');

  let referencePosesLoaded = false;
  let clusterSubTabLoaded = false;

  menuItems.forEach((item) => {
    item.addEventListener('click', () => {
      // remove 'active' from all menu items
      menuItems.forEach(mi => mi.classList.remove('active'));
      // add 'active' to the clicked item
      item.classList.add('active');

      // show/hide the relevant sub-subtab content
      const targetId = item.dataset.tab; // e.g. "reference-handshapes", "cluster-handshapes", or "predict-handshapes"
      contents.forEach((c) => {
        if (c.id === targetId) c.classList.add('active');
        else c.classList.remove('active');
      });
      // if user clicked "See Reference Handshapes" for the first time:
      if (targetId === 'reference-handshapes' && !referencePosesLoaded) {
            initReferencePoseHSViewer();
            referencePosesLoaded = true;
        }
        if (targetId === 'cluster-handshapes' && !clusterSubTabLoaded) {
            console.log('Initializing cluster subtab');
            initClusterSubTab();
            clusterSubTabLoaded = true;
        }
    });
  });

  // 2. Initialize your reference viewer (only needed for Tab #1)
  //    If you want to load the reference pose viewer logic immediately,
  //    you can keep this line. Otherwise, you might wait until
  //    the user clicks the first tab.
  initReferencePoseHSViewer();

  // 3. Remove or comment out the line that auto-clicked the first sub-subtab
  // if (menuItems.length > 0) menuItems[0].click();
}
