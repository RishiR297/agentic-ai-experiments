import sqlite3
import os
import re

def initialize_database(db_path, sql_folder):
    """
    Initialize the SQLite database by executing all SQL files in the given folder.

    Args:
        db_path (str): Path to the SQLite database file.
        sql_folder (str): Path to the folder containing SQL files.
    """
    try:
        # Connect to the SQLite database (creates the file if it doesn't exist)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Preprocess the SQL script to make it compatible with SQLite.
        # Removes schema prefixes like `beautycenterdemo.dbo`.
        def preprocess_sql(sql_script):
            """
            Preprocess the SQL script to make it compatible with SQLite.
            - Removes schema prefixes like `beautycenterdemo.dbo`.
            - Replaces `N'...'` with `'...'`.

            Args:
                sql_script (str): The raw SQL script.

            Returns:
                str: The preprocessed SQL script.
            """
            # Remove schema prefixes (e.g., `beautycenterdemo.dbo.TableName` -> `TableName`)
            sql_script = sql_script.replace("beautycenterdemo.dbo.", "")

            # Replace SQL Server's unicode string prefix N'...'
            # only when it appears as a literal prefix, not inside words
            # like CONSULTATION'.
            sql_script = re.sub(r"(?<![A-Za-z0-9_])N'", "'", sql_script)

            return sql_script

        # Ensure the required tables exist before inserting data.
        # Dynamically creates the `COR_BranchOpeningHour` and `COR_Doctor` tables if they do not exist.
        def ensure_table_exists(cursor):
            """
            Ensure the required tables exist before inserting data.
            Dynamically creates the required tables if they do not exist.
            """
            # Create COR_BranchOpeningHour table if it does not exist
            create_branch_opening_hour_table_query = """
            CREATE TABLE IF NOT EXISTS COR_BranchOpeningHour (
                BranchId INTEGER,
                WeekDay INTEGER,
                OpeningTime TEXT,
                ClosingTime TEXT,
                CreatedOn TEXT,
                CreatedBy TEXT,
                ModifiedOn TEXT,
                ModifiedBy TEXT,
                IsDefault INTEGER,
                IsClosed INTEGER,
                IsActive INTEGER
            );
            """
            cursor.executescript(create_branch_opening_hour_table_query)

            # Create COR_Doctor table if it does not exist
            create_doctor_table_query = """
            CREATE TABLE IF NOT EXISTS COR_Doctor (
                UserId TEXT,
                SpecialtyId INTEGER,
                GlobalId TEXT,
                Firstname TEXT,
                Lastname TEXT,
                DisplayName TEXT,
                Phone TEXT,
                Email TEXT,
                Photo TEXT,
                Signature TEXT,
                SubscriptionId INTEGER,
                DefaultBranchId INTEGER,
                DefaultServiceId INTEGER,
                DefaultVisitDetailId INTEGER,
                IsLicensed INTEGER,
                IsAssistant INTEGER,
                Color TEXT,
                IsLockEnabled INTEGER,
                LockTimeInMinutes INTEGER,
                CreatedOn TEXT,
                CreatedBy TEXT,
                ModifiedOn TEXT,
                ModifiedBy TEXT,
                IsActive INTEGER,
                [Order] INTEGER
            );
            """
            cursor.executescript(create_doctor_table_query)

            # Create COR_DoctorOffSchedule table if it does not exist
            create_doctor_off_schedule_table_query = """
            CREATE TABLE IF NOT EXISTS COR_DoctorOffSchedule (
                DoctorId INTEGER,
                BranchId INTEGER,
                WeekDay INTEGER,
                [Date] TEXT,
                FromTime TEXT,
                ToTime TEXT,
                IsOff INTEGER,
                Reason TEXT,
                CreatedOn TEXT,
                CreatedBy TEXT,
                ModifiedOn TEXT,
                ModifiedBy TEXT,
                IsActive INTEGER
            );
            """
            cursor.executescript(create_doctor_off_schedule_table_query)

            # Create COR_DoctorSchedule table if it does not exist
            create_doctor_schedule_table_query = """
            CREATE TABLE IF NOT EXISTS COR_DoctorSchedule (
                DoctorId INTEGER,
                BranchId INTEGER,
                WeekDay INTEGER,
                FromTime TEXT,
                ToTime TEXT,
                IsOff INTEGER,
                CreatedOn TEXT,
                CreatedBy TEXT,
                ModifiedOn TEXT,
                ModifiedBy TEXT,
                IsActive INTEGER
            );
            """
            cursor.executescript(create_doctor_schedule_table_query)

            # Create View_Appointments table if it does not exist
            create_view_appointments_table_query = """
            CREATE TABLE IF NOT EXISTS View_Appointments (
                AppointmentId INTEGER,
                PatientId INTEGER,
                PatientName TEXT,
                DoctorId INTEGER,
                DoctorName TEXT,
                BranchId INTEGER,
                BranchName TEXT,
                CategoryId INTEGER,
                CategoryName TEXT,
                ServiceId INTEGER,
                ServiceName TEXT,
                MachineId INTEGER,
                MachineName TEXT,
                StartDateTime TEXT,
                EndDateTime TEXT,
                StatusId INTEGER,
                Status TEXT
            );
            """
            cursor.executescript(create_view_appointments_table_query)

            # Create View_Appointments_Setup table if it does not exist
            create_view_appointments_setup_table_query = """
            CREATE TABLE IF NOT EXISTS View_Appointments_Setup (
                ServiceId INTEGER,
                ServiceName TEXT,
                CategoryId INTEGER,
                CategoryName TEXT,
                BranchId INTEGER,
                BranchName TEXT,
                DoctorId INTEGER,
                DoctorName TEXT,
                MachineId INTEGER,
                MachineName TEXT
            );
            """
            cursor.executescript(create_view_appointments_setup_table_query)

        # Ensure tables exist before processing SQL files
        ensure_table_exists(cursor)

        # Iterate over all .sql files in the folder
        for filename in os.listdir(sql_folder):
            if filename.endswith('.sql'):
                file_path = os.path.join(sql_folder, filename)
                print(f"Executing {filename}...")

                # Read and preprocess the SQL file
                with open(file_path, 'r', encoding='utf-8') as sql_file:
                    raw_sql_script = sql_file.read()
                    sql_script = preprocess_sql(raw_sql_script)

                # Execute the preprocessed SQL script
                cursor.executescript(sql_script)

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        print("Database initialized successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the database path and the folder containing SQL files
    db_path = "output.db"  # Change this if you want a different name or location
    sql_folder = os.path.dirname(__file__)  # Current folder

    # Initialize the database
    initialize_database(db_path, sql_folder)
