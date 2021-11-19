#include <Servo.h>

Servo myservo;

String mode="";
int check = 0;
int pos = 0;

void setup() {
  Serial.begin(9600);
  myservo.attach(9);
  myservo.write(pos);
  delay(1000);
}

void loop() {
  Serial.println(":");
  while(Serial.available()==0){
    }
  check = Serial.parseInt();
  Serial.println(check);
  if (check==1){
    for (pos = 0; pos <= 100; pos += 1)
    Serial.println(pos);
    myservo.write(pos); 
    delay(15);
    }
  if (check==2){
    for (pos = 180; pos >= 0; pos -= 1)
    myservo.write(pos); 
    Serial.println(pos);
    delay(5);
    }
}
