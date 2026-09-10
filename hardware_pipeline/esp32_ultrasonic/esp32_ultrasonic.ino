#define TRIG_PIN 5
#define ECHO_PIN 18

long duration;
float distance_cm;

void setup() {
  Serial.begin(115200); 
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

vo
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  
  duration = pulseIn(ECHO_PIN, HIGH);

  
  distance_cm = duration * 0.034 / 2.0;

  
  Serial.println(distance_cm);

  delay(100); 
}
