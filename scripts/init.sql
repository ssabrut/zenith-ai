-- docker/init.sql

-- 1. Patients Table
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    phone_number VARCHAR(20) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Doctors Table
CREATE TABLE IF NOT EXISTS doctors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    phone_number VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE
);

-- 3. Doctor Availability (Schedules)
-- Stores generic weekly availability (e.g., Dr. Andi is available Monday 09:00 - 12:00)
CREATE TABLE IF NOT EXISTS doctor_schedules (
    id SERIAL PRIMARY KEY,
    doctor_id INT REFERENCES doctors(id) ON DELETE CASCADE,
    day_of_week VARCHAR(10) NOT NULL, -- 'Monday', 'Tuesday', etc.
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    UNIQUE(doctor_id, day_of_week, start_time) -- Prevent duplicate slots
);

-- 4. Treatments Table (Services)
CREATE TABLE IF NOT EXISTS treatments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL, -- e.g., 500000.00
    duration_minutes INT DEFAULT 60,
    is_active BOOLEAN DEFAULT TRUE
);

-- 5. Appointments Table (Transactions)
CREATE TABLE IF NOT EXISTS appointments (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id),
    doctor_id INT REFERENCES doctors(id),
    treatment VARCHAR(255),
    appointment_date TIMESTAMP NOT NULL, -- Specific date and time (e.g., 2023-10-25 10:00:00)
    status VARCHAR(20) DEFAULT 'Scheduled', -- 'Scheduled', 'Completed', 'Cancelled', 'No Show'
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Indexes for Performance (Optional but recommended)
CREATE INDEX idx_appointments_date ON appointments(appointment_date);
CREATE INDEX idx_patients_phone ON patients(phone_number);

-- --- SEED DATA (Dummy Data for Testing) ---

INSERT INTO doctors (name, specialization) VALUES 
('Dr. Budi Santoso', '0987654321'),
('Dr. Siti Aminah', '0123456789');

-- Dr. Budi is available
INSERT INTO doctor_schedules (doctor_id, day_of_week, start_time, end_time) VALUES
(1, 'Monday', '09:00', '17:00'),
(1, 'Tuesday', '09:00', '17:00'),
(1, 'Wednesday', '09:00', '17:00'),
(1, 'Friday', '09:00', '17:00');

-- Dr. Siti is available
INSERT INTO doctor_schedules (doctor_id, day_of_week, start_time, end_time) VALUES
(2, 'Thursday', '10:00', '18:00'),
(2, 'Saturday', '10:00', '18:00'),
(2, 'Sunday', '10:00', '18:00');