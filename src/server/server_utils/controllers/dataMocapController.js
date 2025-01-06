// controllers/dataMocapController.js
const fs = require('fs');
const path = require('path');

const uploadDir = path.join(__dirname, '..','..', 'uploads');
const mocapDir = path.join(__dirname, '..','..', 'public', 'data', 'mocap');

// Ensure upload directories exist
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

if (!fs.existsSync(mocapDir)) {
    fs.mkdirSync(mocapDir, { recursive: true });
}

exports.uploadFiles = (req, res) => {
    const uploadedFiles = [];
    const ignoredFiles = [];

    req.files.forEach(file => {
        const targetPath = path.join(mocapDir, file.originalname);

        if (fs.existsSync(targetPath)) {
            console.log(`File already exists, ignoring: ${file.originalname}`);
            ignoredFiles.push(file.originalname);
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

    res.json({
        message: 'Upload completed.',
        uploadedFiles,
        ignoredFiles,
    });
};

exports.listFiles = (req, res) => {
    fs.readdir(mocapDir, (err, files) => {
        if (err) {
            console.error('Error reading MoCap directory:', err);
            return res.status(500).json({ error: 'Failed to fetch file list.' });
        }

        const validExtensions = ['.csv'];
        const filteredFiles = files.filter(file =>
            validExtensions.includes(path.extname(file).toLowerCase())
        );

        res.json(filteredFiles);
    });
};

exports.deleteFiles = (req, res) => {
    const { files } = req.body;

    if (!files || !Array.isArray(files) || files.length === 0) {
        return res.status(400).json({ error: 'No files specified for deletion.' });
    }

    const deletedFiles = [];
    const failedFiles = [];

    files.forEach(file => {
        const filePath = path.join(mocapDir, file);

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

    res.json({
        deletedFiles,
        failedFiles,
        message: `${deletedFiles.length} files deleted, ${failedFiles.length} failed.`,
    });
};
