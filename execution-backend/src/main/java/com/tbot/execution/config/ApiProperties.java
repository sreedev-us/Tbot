package com.tbot.execution.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

@Getter
@Setter
@ConfigurationProperties(prefix = "tbot.api")
public class ApiProperties {

    private String allowedOrigin = "http://localhost:3000";
}
