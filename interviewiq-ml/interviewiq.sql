-- ─────────────────────────────────────────────────────────────
-- InterviewIQ — MySQL Database Setup
-- Run this file in MySQL Workbench or terminal before starting app
-- Command: mysql -u root -p < interviewiq.sql
-- ─────────────────────────────────────────────────────────────

-- Step 1: Create and select the database
CREATE DATABASE IF NOT EXISTS interviewiq;
USE interviewiq;

-- Step 2: Users table (login & registration)
CREATE TABLE IF NOT EXISTS user (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100)  NOT NULL,
    email       VARCHAR(150)  NOT NULL UNIQUE,
    password    VARCHAR(300)  NOT NULL
);

-- Step 3: Interview sessions table
CREATE TABLE IF NOT EXISTS session (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT          NOT NULL,
    job_role        VARCHAR(200),
    predicted_role  VARCHAR(200),
    confidence      FLOAT,
    job_desc        TEXT,
    skills          TEXT,         -- stored as JSON string
    questions       LONGTEXT,     -- stored as JSON string
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(id) ON DELETE CASCADE
);

-- ─────────────────────────────────────────────────────────────
-- Done! Now update app.py with your MySQL credentials and run:
--   python app.py
-- ─────────────────────────────────────────────────────────────