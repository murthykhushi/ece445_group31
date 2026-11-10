#include <Arduino.h>
#include <SPI.h>
#include <TFT_eSPI.h>
#include <ESP32Servo.h>

TFT_eSPI tft = TFT_eSPI();
Servo tailServo;

// Hardware Pins
const int SERVO_PIN      = 13; // 50Hz PWM signal for HS-311 Servo
const int STATUS_LED_PIN = 25; // Existing stand-in LED on GPIO 25
const int DEBUG_PIN      = 26; // Oscilloscope debug timing pin on GPIO 26
const int TRIG_PIN       = 16; // HC-SR04 Trigger pin
const int ECHO_PIN       = 17; // HC-SR04 Echo pin

// HS-311 Non-Blocking Servo Tail State Machine
const int MIN_SAFE_ANGLE          = 25;
const int MAX_SAFE_ANGLE          = 155;
const unsigned long SERVO_TICK_MS = 15;

float currentAngle       = 90.0;
float targetAngle        = 90.0;
float stepDeg            = 3.0;
int wagCyclesRemaining   = 0;
bool wagDirectionUp      = true;
unsigned long lastServoTick = 0;

// HC-SR04 Petting & Score Boost Configuration
const float PET_DISTANCE_CM               = 12.0;
const unsigned long ULTRASONIC_POLL_MS    = 150;
const unsigned long PETTING_HOLD_MS       = 2500;
const int CONSECUTIVE_HITS_REQUIRED       = 2;
const int BOOST_PTS_PER_SEC               = 2;
const int MAX_BOOST_PER_SESSION           = 12;

bool isBeingPetted                = false;
int petHitCount                   = 0;
unsigned long lastUltrasonicPoll  = 0;
unsigned long lastHandDetectedMs  = 0;
unsigned long uncreditedPetMs     = 0;
int sessionBoostTotal             = 0;

// Custom 16-Bit RGB565 Colors
#define COLOR_BG         0x0000 // Pure black background
#define COLOR_CYAN_EYE   0x46FF // Glowing cyan "O w O" face
#define COLOR_CREAM_WARM 0xFFDF // Warm cream/white "^ ^" face
#define COLOR_PINK_BLUSH 0xFB56 // Soft coral/pink oval cheeks & tongue
#define COLOR_AMBER_WARN 0xFD20 // Distracted amber
#define COLOR_RED_ALERT  0xF986 // Highly distracted coral-red

// Spatial Delta-Update Sprites (RAM Buffers)
TFT_eSprite eyeSprite   = TFT_eSprite(&tft); // 110x100 (22 KB) - Reused for Left & Right eyes
TFT_eSprite lowerSprite = TFT_eSprite(&tft); // 320x75  (48 KB) - Holds both blush cheeks + mouth
TFT_eSprite hudSprite   = TFT_eSprite(&tft); // 320x30  (19 KB) - Bottom focus score telemetry

// Screen Coordinates (480x320 Landscape)
const int LEFT_EYE_X  = 95,  LEFT_EYE_Y  = 65;
const int RIGHT_EYE_X = 275, RIGHT_EYE_Y = 65;
const int LOWER_X     = 80,  LOWER_Y     = 170;
const int HUD_X       = 80,  HUD_Y       = 270;

// Dirty Region Bitmask
const uint8_t DIRTY_EYES  = 0x01;
const uint8_t DIRTY_LOWER = 0x02;
const uint8_t DIRTY_HUD   = 0x04;
const uint8_t DIRTY_ALL   = 0x07;

// Four Behavioral States
enum BehavioralState {
  STATE_HIGHLY_FOCUSED    = 0, // S >= 80
  STATE_FOCUSED           = 1, // 60 <= S < 80
  STATE_DISTRACTED        = 2, // 30 <= S < 60
  STATE_HIGHLY_DISTRACTED = 3  // S < 30
};

BehavioralState currentState = STATE_FOCUSED;
int currentScore = 75;

// Non-Blocking Eye Blinking Timers
bool isBlinking = false;
unsigned long lastBlinkTime = 0;
unsigned long nextBlinkInterval = 3500;
const unsigned long BLINK_CLOSED_MS = 110;

// Non-Blocking Stand-In LED State Machine
bool ledPinState = HIGH;
unsigned long lastLedToggle = 0;
int transitionFlashesRemaining = 0;

// SPATIAL DELTA RENDERERS (Off-screen RAM Sprite Drawing)

void renderEyeSprite(BehavioralState state, bool isLeft, bool blinkClosed) {
  eyeSprite.fillSprite(COLOR_BG);

  BehavioralState effectiveState = isBeingPetted ? STATE_HIGHLY_FOCUSED : state;

  if (blinkClosed) {
    uint16_t lidColor = (effectiveState == STATE_FOCUSED)    ? COLOR_CYAN_EYE :
                        (effectiveState == STATE_DISTRACTED) ? COLOR_AMBER_WARN : COLOR_RED_ALERT;
    eyeSprite.fillRoundRect(15, 48, 80, 10, 5, lidColor);
    return;
  }

  switch (effectiveState) {
    case STATE_HIGHLY_FOCUSED:
      // Happy upward-curved "^" arc eyes
      eyeSprite.fillCircle(55, 62, 38, COLOR_CREAM_WARM);
      eyeSprite.fillCircle(55, 78, 30, COLOR_BG);
      eyeSprite.fillRect(0, 68, 110, 32, COLOR_BG);
      eyeSprite.fillCircle(23, 65, 5, COLOR_CREAM_WARM);
      eyeSprite.fillCircle(87, 65, 5, COLOR_CREAM_WARM);
      break;

    case STATE_FOCUSED:
      // Thick glowing Cyan "O" ring eyes
      eyeSprite.fillCircle(55, 50, 38, COLOR_CYAN_EYE);
      eyeSprite.fillCircle(55, 50, 25, COLOR_BG);
      break;

    case STATE_DISTRACTED:
      // Drowsy / Unamused half-lidded eyes (30 <= S < 60)
      eyeSprite.fillCircle(55, 52, 34, COLOR_AMBER_WARN);
      eyeSprite.fillCircle(55, 52, 22, COLOR_BG);
      eyeSprite.fillRect(10, 10, 90, 36, COLOR_BG);
      eyeSprite.fillRoundRect(15, 42, 80, 9, 4, COLOR_AMBER_WARN);
      break;

    case STATE_HIGHLY_DISTRACTED:
      // Stern / Alert angled eyes (S < 30)
      eyeSprite.fillCircle(55, 54, 34, COLOR_RED_ALERT);
      if (isLeft) {
        eyeSprite.fillTriangle(5, 10, 105, 10, 105, 58, COLOR_BG);
        eyeSprite.drawLine(15, 18, 95, 54, COLOR_RED_ALERT);
        eyeSprite.drawLine(15, 19, 95, 55, COLOR_RED_ALERT);
      } else {
        eyeSprite.fillTriangle(5, 10, 105, 10, 5, 58, COLOR_BG);
        eyeSprite.drawLine(95, 18, 15, 54, COLOR_RED_ALERT);
        eyeSprite.drawLine(95, 19, 15, 55, COLOR_RED_ALERT);
      }
      break;
  }
}

void renderLowerFaceSprite(BehavioralState state) {
  lowerSprite.fillSprite(COLOR_BG);

  if (isBeingPetted) {
    // Extra-blushing affectionate open happy mouth when petted
    lowerSprite.fillRoundRect(6, 12, 56, 26, 13, COLOR_PINK_BLUSH);
    lowerSprite.fillRoundRect(258, 12, 56, 26, 13, COLOR_PINK_BLUSH);
    lowerSprite.fillCircle(160, 18, 28, COLOR_PINK_BLUSH);
    lowerSprite.fillRect(125, 0, 70, 18, COLOR_BG);
    return;
  }

  switch (state) {
    case STATE_HIGHLY_FOCUSED:
      // Pink oval cheeks + cheerful open "D" mouth
      lowerSprite.fillRoundRect(10, 15, 48, 22, 11, COLOR_PINK_BLUSH);
      lowerSprite.fillRoundRect(262, 15, 48, 22, 11, COLOR_PINK_BLUSH);
      lowerSprite.fillCircle(160, 18, 28, COLOR_PINK_BLUSH);
      lowerSprite.fillRect(125, 0, 70, 18, COLOR_BG);
      break;

    case STATE_FOCUSED:
      // Pink oval cheeks + cute cat "w" (omega) mouth
      lowerSprite.fillRoundRect(10, 18, 48, 20, 10, COLOR_PINK_BLUSH);
      lowerSprite.fillRoundRect(262, 18, 48, 20, 10, COLOR_PINK_BLUSH);
      lowerSprite.fillCircle(143, 18, 19, COLOR_CYAN_EYE);
      lowerSprite.fillCircle(177, 18, 19, COLOR_CYAN_EYE);
      lowerSprite.fillCircle(143, 15, 13, COLOR_BG);
      lowerSprite.fillCircle(177, 15, 13, COLOR_BG);
      lowerSprite.fillRect(115, 0, 90, 18, COLOR_BG);
      break;

    case STATE_DISTRACTED:
      lowerSprite.fillRoundRect(125, 28, 70, 8, 4, COLOR_AMBER_WARN);
      break;

    case STATE_HIGHLY_DISTRACTED:
      lowerSprite.fillCircle(160, 52, 26, COLOR_RED_ALERT);
      lowerSprite.fillCircle(160, 58, 20, COLOR_BG);
      lowerSprite.fillRect(125, 48, 70, 25, COLOR_BG);
      break;
  }
}

void renderHudSprite(BehavioralState state, int score) {
  hudSprite.fillSprite(COLOR_BG);

  if (isBeingPetted) {
    char petBuf[40];
    snprintf(petBuf, sizeof(petBuf), "PETTING! (+%d PTS) [%d/100]", sessionBoostTotal, score);
    hudSprite.setTextColor(COLOR_PINK_BLUSH, COLOR_BG);
    hudSprite.setTextDatum(MC_DATUM);
    hudSprite.setTextSize(2);
    hudSprite.drawString(petBuf, 160, 15);
    return;
  }

  const char* stateLabel = (state == STATE_HIGHLY_FOCUSED) ? "HIGHLY FOCUSED" :
                           (state == STATE_FOCUSED)        ? "FOCUSED" :
                           (state == STATE_DISTRACTED)     ? "DISTRACTED" : "HIGHLY DISTRACTED";

  uint16_t textColor = (state == STATE_HIGHLY_FOCUSED) ? COLOR_CREAM_WARM :
                       (state == STATE_FOCUSED)        ? COLOR_CYAN_EYE :
                       (state == STATE_DISTRACTED)     ? COLOR_AMBER_WARN : COLOR_RED_ALERT;

  char buf[40];
  snprintf(buf, sizeof(buf), "%s (%d/100)", stateLabel, score);

  hudSprite.setTextColor(textColor, COLOR_BG);
  hudSprite.setTextDatum(MC_DATUM);
  hudSprite.setTextSize(2);
  hudSprite.drawString(buf, 160, 15);
}

void pushDeltaUpdates(uint8_t dirtyMask, bool blinkClosed = false) {
  digitalWrite(DEBUG_PIN, HIGH);

  if (dirtyMask & DIRTY_EYES) {
    renderEyeSprite(currentState, true, blinkClosed);
    eyeSprite.pushSprite(LEFT_EYE_X, LEFT_EYE_Y);

    renderEyeSprite(currentState, false, blinkClosed);
    eyeSprite.pushSprite(RIGHT_EYE_X, RIGHT_EYE_Y);
  }

  if (dirtyMask & DIRTY_LOWER) {
    renderLowerFaceSprite(currentState);
    lowerSprite.pushSprite(LOWER_X, LOWER_Y);
  }

  if (dirtyMask & DIRTY_HUD) {
    renderHudSprite(currentState, currentScore);
    hudSprite.pushSprite(HUD_X, HUD_Y);
  }

  digitalWrite(DEBUG_PIN, LOW);
}

// HS-311 SERVO (GPIO 13), LED (GPIO 25) & HC-SR04 PETTING (GPIO 16/17)

void triggerTailAndLedGesture(BehavioralState newState) {
  lastLedToggle = millis();

  switch (newState) {
    case STATE_HIGHLY_FOCUSED:
      stepDeg = 3.0;
      wagCyclesRemaining = 4;
      wagDirectionUp = true;
      targetAngle = 135.0;
      transitionFlashesRemaining = 6;
      break;

    case STATE_FOCUSED:
      stepDeg = 3.0;
      wagCyclesRemaining = 2;
      wagDirectionUp = true;
      targetAngle = 135.0;
      transitionFlashesRemaining = 2;
      break;

    case STATE_DISTRACTED:
      stepDeg = 1.2;
      wagCyclesRemaining = 0;
      targetAngle = 70.0;
      transitionFlashesRemaining = 0;
      break;

    case STATE_HIGHLY_DISTRACTED:
      stepDeg = 3.0;
      wagCyclesRemaining = 1;
      wagDirectionUp = false;
      targetAngle = 45.0;
      transitionFlashesRemaining = 0;
      break;
  }
}

void updateServoTail() {
  unsigned long now = millis();
  if (now - lastServoTick >= SERVO_TICK_MS) {
    lastServoTick = now;

    float diff = targetAngle - currentAngle;
    if (abs(diff) > 0.5f) {
      if (diff > 0) currentAngle = min(currentAngle + stepDeg, targetAngle);
      else          currentAngle = max(currentAngle - stepDeg, targetAngle);

      int safeAngle = constrain((int)currentAngle, MIN_SAFE_ANGLE, MAX_SAFE_ANGLE);
      tailServo.write(safeAngle);
    } 
    else if (wagCyclesRemaining > 0) {
      wagCyclesRemaining--;
      if (wagCyclesRemaining == 0) {
        targetAngle = isBeingPetted                            ? 120.0 :
                      (currentState == STATE_HIGHLY_FOCUSED)   ? 115.0 :
                      (currentState == STATE_FOCUSED)          ? 100.0 :
                      (currentState == STATE_DISTRACTED)       ? 70.0  : 45.0;
      } else {
        wagDirectionUp = !wagDirectionUp;
        targetAngle = wagDirectionUp ? 135.0 : 45.0;
      }
    }
  }
}

BehavioralState classifyWithHysteresis(int score, BehavioralState prevState) {
  switch (prevState) {
    case STATE_HIGHLY_FOCUSED:
      if (score >= 75) return STATE_HIGHLY_FOCUSED;
      if (score >= 60) return STATE_FOCUSED;
      if (score >= 30) return STATE_DISTRACTED;
      return STATE_HIGHLY_DISTRACTED;

    case STATE_FOCUSED:
      if (score >= 80) return STATE_HIGHLY_FOCUSED;
      if (score >= 55) return STATE_FOCUSED;
      if (score >= 30) return STATE_DISTRACTED;
      return STATE_HIGHLY_DISTRACTED;

    case STATE_DISTRACTED:
      if (score >= 80) return STATE_HIGHLY_FOCUSED;
      if (score >= 60) return STATE_FOCUSED;
      if (score >= 25) return STATE_DISTRACTED;
      return STATE_HIGHLY_DISTRACTED;

    case STATE_HIGHLY_DISTRACTED:
    default:
      if (score >= 80) return STATE_HIGHLY_FOCUSED;
      if (score >= 60) return STATE_FOCUSED;
      if (score >= 35) return STATE_DISTRACTED;
      return STATE_HIGHLY_DISTRACTED;
  }
}

float readUltrasonicDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long durationUs = pulseIn(ECHO_PIN, HIGH, 15000);
  if (durationUs <= 0) {
    return 999.0;
  }
  return (durationUs * 0.0343f) / 2.0f;
}

void applyPetBoostPoints(int pts) {
  if (pts <= 0 || sessionBoostTotal >= MAX_BOOST_PER_SESSION) return;

  int allowedPts = min(pts, MAX_BOOST_PER_SESSION - sessionBoostTotal);
  sessionBoostTotal += allowedPts;
  currentScore = constrain(currentScore + allowedPts, 0, 100);
  currentState = classifyWithHysteresis(currentScore, currentState);

  // Stream upstream telemetry event to live_inference_test.py
  Serial.printf("EVENT:PET_BOOST:%d\n", allowedPts);
  pushDeltaUpdates(DIRTY_HUD, false);
}

void updateUltrasonicPetting() {
  unsigned long now = millis();

  if (isBeingPetted && (now - lastHandDetectedMs >= PETTING_HOLD_MS)) {
    isBeingPetted = false;
    petHitCount = 0;
    uncreditedPetMs = 0;
    sessionBoostTotal = 0;
    triggerTailAndLedGesture(currentState);
    pushDeltaUpdates(DIRTY_ALL, false);
    Serial.println("EVENT:PETTING_ENDED");
    return;
  }

  // Poll HC-SR04 every 150 ms
  if (now - lastUltrasonicPoll >= ULTRASONIC_POLL_MS) {
    unsigned long dt = now - lastUltrasonicPoll;
    lastUltrasonicPoll = now;
    float distCm = readUltrasonicDistanceCm();

    if (distCm > 1.0f && distCm <= PET_DISTANCE_CM) {
      petHitCount++;
      if (petHitCount >= CONSECUTIVE_HITS_REQUIRED) {
        lastHandDetectedMs = now;

        if (!isBeingPetted) {
          isBeingPetted = true;
          sessionBoostTotal = 0;
          uncreditedPetMs = 0;
          
          stepDeg = 3.0;
          wagCyclesRemaining = 4;
          wagDirectionUp = true;
          targetAngle = 135.0;
          transitionFlashesRemaining = 10;
          lastLedToggle = now;

          pushDeltaUpdates(DIRTY_ALL, false);
          applyPetBoostPoints(BOOST_PTS_PER_SEC);
        } else {
          uncreditedPetMs += dt;
          if (uncreditedPetMs >= 1000) {
            uncreditedPetMs -= 1000;
            if (wagCyclesRemaining < 2) wagCyclesRemaining += 2;
            transitionFlashesRemaining = 4;
            applyPetBoostPoints(BOOST_PTS_PER_SEC);
          }
        }
      }
    } else {
      petHitCount = 0;
    }
  }
}

void updateStatusLed() {
  unsigned long now = millis();

  if (transitionFlashesRemaining > 0) {
    if (now - lastLedToggle >= 70) {
      lastLedToggle = now;
      ledPinState = !ledPinState;
      digitalWrite(STATUS_LED_PIN, ledPinState);
      transitionFlashesRemaining--;
      if (transitionFlashesRemaining == 0) {
        ledPinState = HIGH;
        digitalWrite(STATUS_LED_PIN, HIGH);
      }
    }
    return;
  }

  if (isBeingPetted) {
    if (ledPinState != HIGH) {
      ledPinState = HIGH;
      digitalWrite(STATUS_LED_PIN, HIGH);
    }
    return;
  }

  switch (currentState) {
    case STATE_HIGHLY_FOCUSED:
    case STATE_FOCUSED:
      if (ledPinState != HIGH) {
        ledPinState = HIGH;
        digitalWrite(STATUS_LED_PIN, HIGH);
      }
      break;

    case STATE_DISTRACTED:
      if (now - lastLedToggle >= 500) {
        lastLedToggle = now;
        ledPinState = !ledPinState;
        digitalWrite(STATUS_LED_PIN, ledPinState);
      }
      break;

    case STATE_HIGHLY_DISTRACTED:
      if (now - lastLedToggle >= 100) {
        lastLedToggle = now;
        ledPinState = !ledPinState;
        digitalWrite(STATUS_LED_PIN, ledPinState);
      }
      break;
  }
}

void updateEyeBlinking() {
  if (isBeingPetted || currentState == STATE_HIGHLY_FOCUSED) {
    isBlinking = false;
    lastBlinkTime = millis();
    return;
  }

  unsigned long now = millis();
  if (!isBlinking && (now - lastBlinkTime >= nextBlinkInterval)) {
    isBlinking = true;
    lastBlinkTime = now;
    pushDeltaUpdates(DIRTY_EYES, true);
  } 
  else if (isBlinking && (now - lastBlinkTime >= BLINK_CLOSED_MS)) {
    isBlinking = false;
    lastBlinkTime = now;
    nextBlinkInterval = random(2800, 5500);
    pushDeltaUpdates(DIRTY_EYES, false);
  }
}

// TELEMETRY PACKET PARSER

void applyNewStateAndScore(BehavioralState newState, int newScore) {
  bool stateChanged = (newState != currentState);
  bool scoreChanged = (newScore != currentScore);

  currentState = newState;
  currentScore = newScore;

  if (isBeingPetted) {
    if (scoreChanged) pushDeltaUpdates(DIRTY_HUD, false);
    return;
  }

  if (stateChanged) {
    triggerTailAndLedGesture(currentState);
    pushDeltaUpdates(DIRTY_ALL, false);
  } else if (scoreChanged) {
    pushDeltaUpdates(DIRTY_HUD, false);
  }
}

void processSerialPacket(String line) {
  line.trim();

  // Packet -> "$GF,<state_id>,<score>*<hex_checksum>"
  if (line.startsWith("$GF,")) {
    int starIdx = line.indexOf('*');
    if (starIdx != -1) {
      String payload = line.substring(1, starIdx);
      String rxHex   = line.substring(starIdx + 1);
      uint8_t rxChecksum = (uint8_t)strtol(rxHex.c_str(), NULL, 16);

      uint8_t calcChecksum = 0;
      for (size_t i = 0; i < payload.length(); i++) {
        calcChecksum ^= (uint8_t)payload[i];
      }

      if (calcChecksum != rxChecksum) {
        Serial.printf("NACK:CHECKSUM_ERR (calc=%02X, rx=%02X)\n", calcChecksum, rxChecksum);
        return;
      }

      int firstComma  = payload.indexOf(',');
      int secondComma = payload.indexOf(',', firstComma + 1);
      if (firstComma != -1 && secondComma != -1) {
        int stateInt = payload.substring(firstComma + 1, secondComma).toInt();
        int scoreInt = constrain(payload.substring(secondComma + 1).toInt(), 0, 100);
        BehavioralState newState = (BehavioralState)constrain(stateInt, 0, 3);

        applyNewStateAndScore(newState, scoreInt);
        Serial.printf("ACK:STATE_%d_SCORE_%d\n", (int)currentState, currentScore);
        return;
      }
    }
  }

  if (line.startsWith("SCORE:")) {
    int scoreInt = constrain(line.substring(6).toInt(), 0, 100);
    BehavioralState newState = classifyWithHysteresis(scoreInt, currentState);
    applyNewStateAndScore(newState, scoreInt);
    Serial.printf("ACK:SCORE_%d_STATE_%d\n", currentScore, (int)currentState);
    return;
  }

  if (line == "0" || line.equalsIgnoreCase("HIGHLY_FOCUSED")) {
    applyNewStateAndScore(STATE_HIGHLY_FOCUSED, 92);
    Serial.println("ACK:DEMO_HIGHLY_FOCUSED");
  } else if (line == "1" || line.equalsIgnoreCase("FOCUSED")) {
    applyNewStateAndScore(STATE_FOCUSED, 72);
    Serial.println("ACK:DEMO_FOCUSED");
  } else if (line == "2" || line.equalsIgnoreCase("DISTRACTED")) {
    applyNewStateAndScore(STATE_DISTRACTED, 45);
    Serial.println("ACK:DEMO_DISTRACTED");
  } else if (line == "3" || line.equalsIgnoreCase("HIGHLY_DISTRACTED")) {
    applyNewStateAndScore(STATE_HIGHLY_DISTRACTED, 15);
    Serial.println("ACK:DEMO_HIGHLY_DISTRACTED");
  } else if (line.equalsIgnoreCase("PET")) {
    isBeingPetted = true;
    lastHandDetectedMs = millis();
    sessionBoostTotal = 0;
    stepDeg = 3.0;
    wagCyclesRemaining = 4;
    wagDirectionUp = true;
    targetAngle = 135.0;
    transitionFlashesRemaining = 10;
    pushDeltaUpdates(DIRTY_ALL, false);
    applyPetBoostPoints(BOOST_PTS_PER_SEC);
  }
}

void setup() {
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  tailServo.setPeriodHertz(50);
  tailServo.write((int)currentAngle);
  tailServo.attach(SERVO_PIN, 600, 2400);

  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, HIGH);

  pinMode(DEBUG_PIN, OUTPUT);
  digitalWrite(DEBUG_PIN, LOW);

  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(ECHO_PIN, INPUT);

  tft.init();
  tft.setRotation(3);
  tft.fillScreen(COLOR_BG);

  eyeSprite.setColorDepth(16);
  eyeSprite.createSprite(110, 100);

  lowerSprite.setColorDepth(16);
  lowerSprite.createSprite(320, 75);

  hudSprite.setColorDepth(16);
  hudSprite.createSprite(320, 30);

  pushDeltaUpdates(DIRTY_ALL, false);
  Serial.println("SYS:GROUP31_SERVO_ULTRASONIC_LED_READY");
}

void loop() {
  if (Serial.available() > 0) {
    String incoming = Serial.readStringUntil('\n');
    if (incoming.length() > 0) {
      processSerialPacket(incoming);
    }
  }

  updateUltrasonicPetting();
  updateServoTail();
  updateStatusLed();
  updateEyeBlinking();
}