const express = require('express');
const router = express.Router();
const clusterController = require('../controllers/clusterController');

// POST /api/cluster/handshapes
router.post('/handshapes', clusterController.clusterHandshapes);

module.exports = router;
