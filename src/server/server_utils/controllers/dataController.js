// server_utils/controllers/dataController.js
const fs = require('fs');
const path = require('path');

// Helper function to ensure the subfolder (e.g., “mocap” or “video”) exists
function getTypeDirectory(type) {
  // E.g., type might be "mocap" or "video"
  const dir = path.join(__dirname, '..', '..', 'public', 'data', type);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  return dir;
}

/**
 * Handle file uploads.
 * - Uses :type from the route to determine subfolder
 * - Moves files from temp dir to the correct subfolder
 */
exports.uploadFiles = (req, res) => {
  const { type } = req.params;   // e.g. "mocap" or "video"
  const typeDir = getTypeDirectory(type);

  const uploadedFiles = [];
  const ignoredFiles = [];

  // req.files comes from multer (the array('files', 10) in dataRoutes.js)
  req.files.forEach(file => {
    const targetPath = path.join(typeDir, file.originalname);

    if (fs.existsSync(targetPath)) {
      //console.log(`File already exists, ignoring: ${file.originalname}`);
      ignoredFiles.push(file.originalname);

      // Remove the temporary file
      fs.unlinkSync(file.path);
    } else {
      try {
        fs.renameSync(file.path, targetPath);
        console.log(`File uploaded successfully: ${file.originalname}`);
        uploadedFiles.push({ filename: file.originalname });
      } catch (err) {
        console.error(`Error moving file ${file.originalname}:`, err);
      }
    }
  });

  return res.json({
    message: 'Upload completed.',
    uploadedFiles,
    ignoredFiles
  });
};

/**
 * List files in the chosen subfolder.
 * - GET /api/data/:type
 */
exports.listFiles = (req, res) => {
  const { type } = req.params;
  const typeDir = getTypeDirectory(type);

  fs.readdir(typeDir, (err, files) => {
    if (err) {
      console.error('Error reading directory:', err);
      return res.status(500).json({ error: 'Failed to fetch file list.' });
    }

    // If you want to filter by extension, do it here. For example:
    // if (type === 'mocap') {
    //   files = files.filter(file => path.extname(file).toLowerCase() === '.csv');
    // }
    // else if (type === 'video') {
    //   files = files.filter(file => {
    //     const ext = path.extname(file).toLowerCase();
    //     return ['.mp4', '.mov', '.avi'].includes(ext);
    //   });
    // }

    return res.json(files);
  });
};

/**
 * Delete files from the chosen subfolder.
 * - DELETE /api/data/:type
 */
exports.deleteFiles = (req, res) => {
  const { type } = req.params;
  const { files } = req.body;
  const typeDir = getTypeDirectory(type);

  if (!files || !Array.isArray(files) || files.length === 0) {
    return res.status(400).json({ error: 'No files specified for deletion.' });
  }

  const deletedFiles = [];
  const failedFiles = [];

  files.forEach(file => {
    const filePath = path.join(typeDir, file);

    if (fs.existsSync(filePath)) {
      try {
        fs.unlinkSync(filePath);
        deletedFiles.push(file);
      } catch (err) {
        console.error(`Error deleting file ${file}:`, err);
        failedFiles.push(file);
      }
    } else {
      failedFiles.push(file);
    }
  });

  return res.json({
    deletedFiles,
    failedFiles,
    message: `${deletedFiles.length} files deleted, ${failedFiles.length} failed.`
  });
};
