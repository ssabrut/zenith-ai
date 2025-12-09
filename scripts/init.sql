-- docker/init.sql

-- 1. Patients Table
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    phone_number VARCHAR(20) UNIQUE NOT NULL,
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Doctors Table
CREATE TABLE IF NOT EXISTS doctors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    specialization VARCHAR(100),
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
    treatment_id INT REFERENCES treatments(id),
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
('Dr. Budi Santoso', 'Dermatologist'),
('Dr. Siti Aminah', 'Aesthetic Physician');

-- Dr. Budi is available Mon & Wed 9-17
INSERT INTO doctor_schedules (doctor_id, day_of_week, start_time, end_time) VALUES
(1, 'Monday', '09:00', '17:00'),
(1, 'Wednesday', '09:00', '17:00');

-- Dr. Siti is available Tue & Thu 10-18
INSERT INTO doctor_schedules (doctor_id, day_of_week, start_time, end_time) VALUES
(2, 'Tuesday', '10:00', '18:00'),
(2, 'Thursday', '10:00', '18:00');

INSERT INTO treatments (name, price, duration_minutes) VALUES
('Acne Facial', 350000, 60),
('Laser Rejuvenation', 1500000, 45),
('Chemical Peeling', 500000, 30);

INSERT INTO patients (full_name, phone_number) VALUES
('Andi Pratama', '081234567890');