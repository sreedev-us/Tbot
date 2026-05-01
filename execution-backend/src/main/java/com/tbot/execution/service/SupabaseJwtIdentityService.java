package com.tbot.execution.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Map;
import org.springframework.stereotype.Service;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

@Service
public class SupabaseJwtIdentityService {

    private static final TypeReference<Map<String, Object>> CLAIM_MAP = new TypeReference<>() {};

    private final ObjectMapper objectMapper;

    public SupabaseJwtIdentityService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public String extractUserId(String authorizationHeader) {
        if (authorizationHeader == null || authorizationHeader.isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Missing Authorization header.");
        }
        if (!authorizationHeader.startsWith("Bearer ")) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Authorization header must be a Bearer token.");
        }

        String token = authorizationHeader.substring("Bearer ".length()).trim();
        String[] segments = token.split("\\.");
        if (segments.length < 2) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Malformed JWT.");
        }

        try {
            byte[] payloadBytes = Base64.getUrlDecoder().decode(segments[1]);
            Map<String, Object> claims = objectMapper.readValue(
                    new String(payloadBytes, StandardCharsets.UTF_8),
                    CLAIM_MAP
            );
            Object subject = claims.get("sub");
            if (subject == null || subject.toString().isBlank()) {
                throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "JWT subject claim is missing.");
            }
            return subject.toString();
        } catch (ResponseStatusException exception) {
            throw exception;
        } catch (Exception exception) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Unable to decode JWT subject.", exception);
        }
    }
}
