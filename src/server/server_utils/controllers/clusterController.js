// clusterController.js
const { spawn } = require('child_process');
const path = require('path');

exports.clusterHandshapes = (req, res) => {
  // 1) Extract data from the request body
  const { dataType, files, precropped, k, visualize } = req.body;

  // Validate if you want
  if (!dataType || !files || !Array.isArray(files) || files.length === 0) {
    return res.status(400).json({ error: 'Missing or invalid data.' });
  }

  // 2) Convert 'files' array to a string or pass them as multiple arguments
  //    For example, pass them as a single comma-separated string:
  const filesArg = files.join(',');

  // 3) Build the arguments array for Python
  //    e.g. script.py dataType k precropped 
  //    You can structure them however you like.
  const scriptPath = path.resolve(__dirname, '../../../modules/handshapes/utils/process_clustering.py');
  
  const args = [
    scriptPath, 
    dataType,
    String(k),
    precropped ? 'true' : 'false',
    filesArg,
    visualize ? 'true' : 'false'
  ];

  console.log('Spawning Python script with args:', args);

  // 4) Spawn the Python process
  const pythonProcess = spawn('python3', args);
  console.log('Python process spawned.');

  // 5) Capture output (stdout & stderr)
  let pythonOutput = '';
  pythonProcess.stdout.on('data', (data) => {
    pythonOutput += data.toString();
  });

  let pythonError = '';
  pythonProcess.stderr.on('data', (data) => {
    pythonError += data.toString();
  });

  // 6) On exit, parse the Python output and return JSON
  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      console.error('Python script exited with error code:', code, pythonError);
      return res.status(500).json({
        error: `Python script error (code ${code}): ${pythonError}`
      });
    }

    // Suppose your Python script prints JSON. Then parse it:
    let parsed;
    try {
      parsed = JSON.parse(pythonOutput);
    } catch (parseErr) {
      console.error('Failed to parse Python output as JSON:', parseErr, pythonOutput);
      return res.status(500).json({ error: 'Invalid JSON output from Python.' });
    }

    // Return the data from Python
    res.json({
      success: true,
      message: 'Clustering completed via Python script.',
      data: parsed
    });
  });
  console.log('Completed.');
};
