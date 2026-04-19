package com.tbot.execution;

import com.tbot.execution.config.ApiProperties;
import com.tbot.execution.config.RiskProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties({RiskProperties.class, ApiProperties.class})
public class ExecutionBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(ExecutionBackendApplication.class, args);
	}

}
