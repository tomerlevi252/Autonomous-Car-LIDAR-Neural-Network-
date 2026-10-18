"""
main.py
=======
סקריפט ראשי של פרויקט CAR ML  —  גרסה V4 (100% רשת נוירונים; נתוני האימון במאגר car_routes.db).

דרייברים:
    [v] motor_driver_L298n.py      — L298N
    [v] display_driver_ILI9341.py  — ILI9341
    [v] touch_driver_XPT2046.py    — XPT2046
    [v] bluetooth_driver.py        — Bluetooth
    [v] lidar_driver.py            — LIDAR D200
    [v] ui.py                      — UI
    [v] route_database.py          — SQLite route recording (car_routes.db)
    [v] auto_drive.py              — CAR RUN (NAV_MODE='ml' = the neural network drives)

למידת מכונה (רשת נוירונים) — לפי ה-PDF:
    ml_features.py   — סריקת LIDAR -> 16 מספרים 0..1 (הקלט של הרשת)
    ml_dataset.py    — המאגר car_routes.db -> טבלת אימון (LIDAR -> מה הנהג לחץ)
    ml_model.py      — רשתות הנוירונים (MLP 32->16, ensemble של 3, NumPy)
    ml_train.py      — אימון + בדיקה (confusion matrix) -> ml_model.npz   (כפתור TRAIN NETWORK)
    ml_driver.py     — הרשת נוהגת בזמן אמת, 100% רשת נוירונים (V3)   (כפתור CAR RUN)
    ml_simulate.py   — נסיעת מבחן וירטואלית של הרשת על מפת מסלול מוקלט
    ml_lstm.py       — ניסוי LSTM (PDF סעיף 10)

הפעלה:
    sudo python3 main.py

עצירה:
    Ctrl+C
"""

import RPi.GPIO as GPIO
from motor_driver_L298n     import MotorDriverL298n
from display_driver_ILI9341 import DisplayDriverILI9341
from touch_driver_XPT2046   import TouchDriverXPT2046
from bluetooth_driver       import BluetoothDriver
from lidar_driver           import LidarDriver
from route_database         import RouteDatabase
from auto_drive             import AutoDrive
from ui                     import UI


def main():
    print("=" * 40)
    print("   CAR ML — מערכת מופעלת")
    print("=" * 40)

    GPIO.setmode(GPIO.BCM)

    # ── אתחול דרייברים ────────────────────────────────────────
    motors    = MotorDriverL298n(default_speed=100)
    display   = DisplayDriverILI9341()
    touch     = TouchDriverXPT2046()
    bluetooth = BluetoothDriver(motors)
    lidar     = LidarDriver(angle_offset=0.0)  # כייל כאן אם צריך
    route_db  = RouteDatabase("car_routes.db")  # ריצה ראשונה — יוצר את הקובץ/הטבלאות לבד
    # CAR RUN — רשת הנוירונים המאומנת (ml_model.npz) נוהגת לפי מה שה-LIDAR רואה.
    # (auto_drive.NAV_MODE = 'lidar' מחזיר את הניווט הידני הישן, להשוואה)
    auto_drive = AutoDrive(motors, route_db, lidar)
    ui        = UI(display, touch)

    ui.set_bluetooth_driver(bluetooth)
    ui.set_lidar_driver(lidar)
    ui.set_route_database(route_db)
    ui.set_auto_drive(auto_drive)

    # פותחים את חיבור ה-UART מיד, אבל עוצרים את המנוע —
    # הוא יתחיל לסובב רק כשנכנסים למסך ה-LIDAR/ML (ui.py קורא ל-start_motor()).
    lidar.start()
    lidar.stop_motor()

    print("[MAIN] כל הדרייברים אותחלו בהצלחה.")
    print("-" * 40)

    try:
        # ui.run() היא הלולאה הראשית — לא חוזרת עד Ctrl+C
        ui.run()

    except KeyboardInterrupt:
        print("\n[MAIN] עצור על ידי המשתמש.")

    finally:
        bluetooth.cleanup()
        lidar.cleanup()
        motors.cleanup()
        display.cleanup()
        touch.cleanup()
        GPIO.cleanup()
        print("[MAIN] המערכת כובתה בצורה תקינה.")


if __name__ == "__main__":
    main()
