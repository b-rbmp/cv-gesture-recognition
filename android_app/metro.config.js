const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

config.resolver.assetExts.push(
  // Adds support for `.tflite` files for TensorFlow Lite models
  'tflite'
);

module.exports = config;
