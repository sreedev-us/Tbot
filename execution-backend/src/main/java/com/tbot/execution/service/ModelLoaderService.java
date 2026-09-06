package com.tbot.execution.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@Service
public class ModelLoaderService {

    @Value("${tbot.models.path:./models}")
    private String modelsBasePath;

    private volatile Map<String, Map<String, Object>> loadedModels = new HashMap<>();
    private volatile Map<String, String> modelHashes = new HashMap<>();

    public Map<String, Object> loadModel(String modelName, String modelVersion) {
        String modelKey = modelName + ":" + modelVersion;

        if (loadedModels.containsKey(modelKey)) {
            log.debug("Model already loaded: {}", modelKey);
            return loadedModels.get(modelKey);
        }

        try {
            Path modelPath = Paths.get(modelsBasePath, modelName, modelVersion);
            File modelDir = modelPath.toFile();

            if (!modelDir.exists()) {
                log.warn("Model directory not found: {}", modelPath);
                return createPlaceholderModel();
            }

            // Load model metadata
            Map<String, Object> model = new HashMap<>();
            model.put("name", modelName);
            model.put("version", modelVersion);
            model.put("loadedAt", System.currentTimeMillis());
            model.put("path", modelPath.toString());

            // Calculate model hash for reproducibility
            String modelHash = calculateModelHash(modelDir);
            model.put("hash", modelHash);
            modelHashes.put(modelKey, modelHash);

            // TODO: Load actual model weights (XGBoost, LightGBM, ONNX, etc.)
            // For now, store placeholder metadata
            model.put("type", "placeholder");
            model.put("featureCount", 0);
            model.put("framework", "unknown");

            loadedModels.put(modelKey, model);
            log.info("Model loaded: {} (hash: {})", modelKey, modelHash);

            return model;
        } catch (Exception e) {
            log.error("Error loading model: {}", modelKey, e);
            return createPlaceholderModel();
        }
    }

    public String getModelHash(String modelName, String modelVersion) {
        String modelKey = modelName + ":" + modelVersion;
        return modelHashes.getOrDefault(modelKey, "unknown");
    }

    public boolean isModelLoaded(String modelName, String modelVersion) {
        return loadedModels.containsKey(modelName + ":" + modelVersion);
    }

    public void unloadModel(String modelName, String modelVersion) {
        String modelKey = modelName + ":" + modelVersion;
        loadedModels.remove(modelKey);
        modelHashes.remove(modelKey);
        log.info("Model unloaded: {}", modelKey);
    }

    private String calculateModelHash(File modelDir) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");

        if (!modelDir.exists()) {
            return "unknown";
        }

        // Calculate hash of all files in model directory
        File[] files = modelDir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isFile()) {
                    byte[] fileBytes = Files.readAllBytes(file.toPath());
                    digest.update(fileBytes);
                }
            }
        }

        byte[] hashBytes = digest.digest();
        StringBuilder hexString = new StringBuilder();
        for (byte b : hashBytes) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) hexString.append('0');
            hexString.append(hex);
        }
        return hexString.toString();
    }

    private Map<String, Object> createPlaceholderModel() {
        Map<String, Object> placeholder = new HashMap<>();
        placeholder.put("type", "placeholder");
        placeholder.put("status", "not_found");
        return placeholder;
    }
}
