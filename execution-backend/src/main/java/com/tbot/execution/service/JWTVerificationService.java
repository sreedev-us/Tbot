package com.tbot.execution.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.nio.charset.StandardCharsets;
import java.security.KeyFactory;
import java.security.PublicKey;
import java.security.Signature;
import java.util.Base64;
import java.util.Map;

@Slf4j
@Service
public class JWTVerificationService {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${supabase.jwt-secret:}")
    private String jwtSecret;

    @Value("${supabase.public-key:}")
    private String publicKeyPem;

    public Map<String, Object> verifyAndDecodeJWT(String token) throws Exception {
        if (token == null || token.isEmpty()) {
            throw new IllegalArgumentException("JWT token is empty");
        }

        String[] parts = token.split("\\.");
        if (parts.length != 3) {
            throw new IllegalArgumentException("Invalid JWT format: expected 3 parts");
        }

        String headerB64 = parts[0];
        String payloadB64 = parts[1];
        String signatureB64 = parts[2];

        // Decode header
        String headerJson = new String(Base64.getUrlDecoder().decode(headerB64), StandardCharsets.UTF_8);
        JsonNode header = objectMapper.readTree(headerJson);

        // Decode payload
        String payloadJson = new String(Base64.getUrlDecoder().decode(payloadB64), StandardCharsets.UTF_8);
        JsonNode payload = objectMapper.readTree(payloadJson);

        // Verify signature
        String signingInput = headerB64 + "." + payloadB64;
        byte[] signature = Base64.getUrlDecoder().decode(signatureB64);

        if (!publicKeyPem.isEmpty()) {
            verifySignatureWithPublicKey(signingInput, signature);
        } else if (!jwtSecret.isEmpty()) {
            verifySignatureWithSecret(signingInput, signature);
        } else {
            log.warn("No JWT verification key configured, skipping signature verification");
        }

        // Convert payload to map
        Map<String, Object> claims = objectMapper.convertValue(payload, Map.class);
        log.debug("JWT verified successfully for subject: {}", claims.get("sub"));

        return claims;
    }

    private void verifySignatureWithPublicKey(String signingInput, byte[] signature) throws Exception {
        PublicKey publicKey = loadPublicKey(publicKeyPem);
        
        Signature sig = Signature.getInstance("SHA256withRSA");
        sig.initVerify(publicKey);
        sig.update(signingInput.getBytes(StandardCharsets.UTF_8));

        if (!sig.verify(signature)) {
            throw new SecurityException("JWT signature verification failed");
        }

        log.debug("JWT signature verified with RSA public key");
    }

    private void verifySignatureWithSecret(String signingInput, byte[] signature) throws Exception {
        javax.crypto.Mac mac = javax.crypto.Mac.getInstance("HmacSHA256");
        javax.crypto.spec.SecretKeySpec secretKey = 
            new javax.crypto.spec.SecretKeySpec(
                jwtSecret.getBytes(StandardCharsets.UTF_8),
                0,
                jwtSecret.getBytes(StandardCharsets.UTF_8).length,
                "HmacSHA256"
            );
        mac.init(secretKey);

        byte[] expectedSignature = mac.doFinal(signingInput.getBytes(StandardCharsets.UTF_8));

        if (!constantTimeEquals(expectedSignature, signature)) {
            throw new SecurityException("JWT signature verification failed");
        }

        log.debug("JWT signature verified with HMAC secret");
    }

    private PublicKey loadPublicKey(String pemString) throws Exception {
        String publicKeyPEM = pemString
            .replace("-----BEGIN PUBLIC KEY-----", "")
            .replace("-----END PUBLIC KEY-----", "")
            .replaceAll("\\s", "");

        byte[] decodedKey = Base64.getDecoder().decode(publicKeyPEM);
        java.security.spec.X509EncodedKeySpec spec = new java.security.spec.X509EncodedKeySpec(decodedKey);
        KeyFactory kf = KeyFactory.getInstance("RSA");
        return kf.generatePublic(spec);
    }

    private boolean constantTimeEquals(byte[] a, byte[] b) {
        if (a.length != b.length) {
            return false;
        }
        int result = 0;
        for (int i = 0; i < a.length; i++) {
            result |= a[i] ^ b[i];
        }
        return result == 0;
    }

    public String extractUserId(String token) throws Exception {
        Map<String, Object> claims = verifyAndDecodeJWT(token);
        Object sub = claims.get("sub");
        if (sub == null) {
            throw new SecurityException("JWT does not contain 'sub' claim");
        }
        return sub.toString();
    }
}
