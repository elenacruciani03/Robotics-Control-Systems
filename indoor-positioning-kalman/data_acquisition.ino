
// Pin sensor 1
const int TRIG1 = 12, ECHO1 = 13;
// Pin sensor 2
const int TRIG2 = 3, ECHO2 = 2;
// Pin sensor 3
const int TRIG3 = 9, ECHO3 = 10;

const int LOOP_DELAY = 200; 

void setup() {ì
  Serial.begin(115200);

  pinMode(TRIG1, OUTPUT);
  pinMode(ECHO1, INPUT);
  pinMode(TRIG2, OUTPUT);
  pinMode(ECHO2, INPUT);
  pinMode(TRIG3, OUTPUT);
  pinMode(ECHO3, INPUT);

  digitalWrite(TRIG1, LOW);
  digitalWrite(TRIG2, LOW);
  digitalWrite(TRIG3, LOW);

}

void loop() {
  float d1 = misura(TRIG1, ECHO1);
  delay(30);
  float d2 = misura(TRIG2, ECHO2);
  delay(30);
  float d3 = misura(TRIG3, ECHO3);

  Serial.print(d1);
  Serial.print(",");
  Serial.print(d2);
  Serial.print(",");
  Serial.println(d3);
  
  delay(LOOP_DELAY);
}

float misura(int trig, int echo) {

  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  
  long durata = pulseIn(echo, HIGH, 10000);
  
  if (durata == 0) {
    return 0.0;
  }

  float distanza = durata*0.0343 / 2;

  return distanza;

}
