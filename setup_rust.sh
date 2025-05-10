#!/bin/bash

# ติดตั้ง Rust toolchain และตั้งค่าเป็น stable
echo "กำลังตั้งค่า Rust toolchain เป็น stable..."
rustup default stable

# ตรวจสอบว่าการติดตั้งสำเร็จหรือไม่
if [ $? -eq 0 ]; then
  echo "✅ การตั้งค่า Rust toolchain สำเร็จ"
  
  # แสดงข้อมูลเวอร์ชันของ Rust และ Cargo
  echo ""
  echo "ข้อมูล Rust toolchain:"
  rustc --version
  cargo --version
else
  echo "❌ เกิดข้อผิดพลาดในการตั้งค่า Rust toolchain"
  exit 1
fi

# ตรวจสอบการคอมไพล์ Luma
echo ""
echo "กำลังทดสอบการคอมไพล์ Luma..."
cd Luma && cargo check

if [ $? -eq 0 ]; then
  echo "✅ การทดสอบการคอมไพล์ Luma สำเร็จ"
  echo "คุณสามารถรันคำสั่งต่อไปนี้ได้แล้ว:"
  echo "  1. cargo run --bin luma -- --repl   # เริ่ม Luma REPL"
  echo "  2. cargo test                       # รัน unit tests"
else
  echo "❌ เกิดข้อผิดพลาดในการคอมไพล์ Luma"
  exit 1
fi

echo ""
echo "การตั้งค่าเสร็จสมบูรณ์!"