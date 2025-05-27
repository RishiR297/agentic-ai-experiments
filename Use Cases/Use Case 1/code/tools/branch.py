from utils.db import get_db_connection

BRANCH_NAMES = {
    1: "Dbayeh",
    2: "Unmapped Branch"
}

def list_branch_opening_hours():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT BranchId, WeekDay, OpeningTime, ClosingTime, IsClosed
        FROM COR_BranchOpeningHour
        WHERE IsActive = 1
        ORDER BY BranchId, WeekDay
    """)
    results = cursor.fetchall()
    conn.close()

    readable_hours = []
    for row in results:
        branch_name = BRANCH_NAMES.get(row[0], f"Branch {row[0]}")
        weekday = weekday_name(row[1])
        if row[4]:  # IsClosed
            time_range = "Closed"
        else:
            # Trim .0000000 from SQL Server-style times
            open_time = row[2].split('.')[0]
            close_time = row[3].split('.')[0]
            time_range = f"{open_time} - {close_time}"
        readable_hours.append((branch_name, weekday, time_range))

    return readable_hours

def weekday_name(index):
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return names[index] if 0 <= index < len(names) else f"Invalid Day {index}"

if __name__ == "__main__":
    print("Branch Opening Hours:")
    for branch_name, weekday, hours in list_branch_opening_hours():
        print(f"{branch_name} | {weekday}: {hours}")
