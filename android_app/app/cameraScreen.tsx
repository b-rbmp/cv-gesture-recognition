import React, { useEffect } from 'react';
import { useState } from 'react';
import { ActivityIndicator, Button, StyleSheet, Text, TouchableOpacity, View, ViewProps } from 'react-native';
import { useTensorflowModel } from 'react-native-fast-tflite';
import Animated, { runOnJS, useAnimatedReaction, useDerivedValue } from 'react-native-reanimated';
import { Camera, CameraDevice, runAtTargetFps, useCameraDevice, useCameraPermission, useFrameProcessor } from 'react-native-vision-camera';
import { useSharedValue } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';

// Label Dictionary for the model
const modelLabelsDict: { [key: string]: string } = {
  "0": 'call',
  "1": 'dislike',
  "2": 'fist',
  "3": 'four',
  "4": 'like',
  "5": 'mute',
  "6": 'no_gesture',
  "7": 'ok',
  "8": 'one',
  "9": 'palm',
  "10": 'peace',
  "11": 'peace_inverted',
  "12": 'rock',
  "13": 'stop',
  "14": 'stop_inverted',
  "15": 'three',
  "16": 'three2',
  "17": 'two_up',
  "18": 'two_up_inverted',
};

type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array

export default function CameraScreen() {
  const plugin = useTensorflowModel(require('../assets/model/model.tflite'));
  const model = plugin.state === "loaded" ? plugin.model : undefined;

  const { resize } = useResizePlugin();
  const frontDevice = useCameraDevice('front');
  const backDevice = useCameraDevice('back');
  const [device, setDevice] = useState<CameraDevice | undefined>(useCameraDevice('front'));
  const facing = useSharedValue('front');
  const { hasPermission, requestPermission } = useCameraPermission()
  const playing = useSharedValue(false);
  const [nonSharedPlaying, setNonSharedPlaying] = useState(false);

  const TARGET_FPS = 3;

  const gameScore = useSharedValue(0);
  const round = useSharedValue(1);
  const maximumRounds = 5;
  const roundTime = useSharedValue(4);
  const roundTimeLimit = 4; // 4 seconds per round
  const lastComputerMove = useSharedValue<null | string>(null);
  const lastPlayerMove = useSharedValue<null | string>(null);
  const lastPredictedMove = useSharedValue<null | string>(null);

  const lastWinner = useSharedValue<null | string>(null);

  // State variables for UI updates
  const [currentRound, setCurrentRound] = useState(1);
  const [currentScore, setCurrentScore] = useState(0);
  const [currentRoundTime, setCurrentRoundTime] = useState(4);
  const [currentLastComputerMove, setCurrentLastComputerMove] = useState<null | string>(null);
  const [currentLastPlayerMove, setCurrentLastPlayerMove] = useState<null | string>(null);
  const [currentLastWinner, setCurrentLastWinner] = useState<null | string>(null);
  const [currentLastPredictedMove, setCurrentLastPredictedMove] = useState<null | string>(null);
  
  
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentRound(round.value);
      setCurrentScore(gameScore.value);
      setCurrentRoundTime(roundTime.value);
      setCurrentLastComputerMove(lastComputerMove.value);
      setCurrentLastPlayerMove(lastPlayerMove.value);
      setCurrentLastWinner(lastWinner.value);
      setNonSharedPlaying(playing.value);
      setCurrentLastPredictedMove(lastPredictedMove.value);
    }, 50);

    return () => clearInterval(interval);
  }, []);

  const gameLogic = (prediction: string) => {
    'worklet'
    if (playing.value) {
      // Increment the round time
      roundTime.value -= 1 / TARGET_FPS;

      lastPredictedMove.value = prediction;
  
      // Check if the round time limit has been reached
      if (roundTime.value <= 0) {
        // Generate a random move for the computer between rock, paper, and scissors
        const computerMove = ['rock', 'paper', 'scissors'][Math.floor(Math.random() * 3)];
        lastComputerMove.value = computerMove;

        lastPlayerMove.value = prediction;

        // Determine the winner of the round
        if (prediction === computerMove) {
          // Draw
        } else if (
          (prediction === 'rock' && computerMove === 'scissors') ||
          (prediction === 'paper' && computerMove === 'rock') ||
          (prediction === 'scissors' && computerMove === 'paper')
        ) {
          // Player wins
          gameScore.value += 1;
        } else {
          // Computer wins
          gameScore.value -= 1;
        }

        // Increment the round
        round.value += 1;
        // Reset the round time
        roundTime.value = roundTimeLimit;
      }

      if (round.value > maximumRounds) {
        // Stop the game
        playing.value = false;

        if (gameScore.value > 0) {
          lastWinner.value = 'Player';
        } else if (gameScore.value < 0) {
          lastWinner.value = 'Computer';
        } else {
          lastWinner.value = 'Draw';
        }
      }
      
    }
  }

  // Function to rearrange the array from (300 * 300 * 3) to (3 * 300 * 300)
  const rearrangeArray = (array: Float32Array, width: number, height: number, channels: number) => {
    'worklet'
    const newArray = new Float32Array(channels * width * height);
  
    for (let i = 0; i < width * height; i++) {
      for (let c = 0; c < channels; c++) {
        newArray[c * width * height + i] = array[i * channels + c];
      }
    }
  
    return newArray;
  };



  const convertPredictionToRPS = (prediction: string) => {
    'worklet'
    const paper = ["palm", "stop", "stop_inverted"];
    const rock = ["fist"];
    const scissors = ["peace", "peace_inverted"];
    let convertedPrediction = null;
  
    // Convert the player move to rock, paper, or scissors
    if (rock.includes(prediction)) {
      convertedPrediction = "rock";
    } else if (paper.includes(prediction)) {
      convertedPrediction = "paper";
    } else if (scissors.includes(prediction)) {
      convertedPrediction = "scissors";
    } else {
      convertedPrediction = "Not Valid";
    }
  
    return convertedPrediction;
  }

  // Function to get the prediction from the model outputs
  // Format of outputs: [{"0": -0.12327446043491364, "1": -0.06530726701021194, "10": 0.008085541427135468, "11": 0.0030853217467665672, "12": -0.036857232451438904, "13": 0.01642770692706108, "14": 0.03786261007189751, "15": 0.011143926531076431, "16": 0.0050139110535383224, "17": 0.06414322555065155, "18": 0.0390976220369339, "2": 0.07954569905996323, "3": -0.0023438120260834694, "4": -0.02880370430648327, "5": -0.11453615128993988, "6": 0.2568812966346741, "7": -0.10927405208349228, "8": 0.05309825390577316, "9": 0.006750617641955614}]
  const getPrediction = (outputs : TypedArray[]) => {
    'worklet'
    const prediction = outputs[0];
  
    let maxKey = null;
    let maxValue = -Infinity;
  
    // Iterate over the prediction object to find the key with the highest value
    for (const key in prediction) {
      if (prediction[key] > maxValue) {
        maxValue = prediction[key] as number;
        maxKey = key;
      }
    }
  
    // Map the key with the highest value to its corresponding label
    const label = modelLabelsDict[maxKey as string];
  
    return label;
  }

  
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet'
    if (model == null) return;


    runAtTargetFps(TARGET_FPS, () => {
      'worklet'
      if (playing.value) {
        // Resize the frame
        const resized = resize(frame, {
          scale: {
            width: 300,
            height: 300,
          },
          pixelFormat: 'rgb',
          dataType: 'float32',
          rotation: facing.value === 'front' ? '270deg' : '90deg',
        });
        
        // Convert resized which is Float32Array = (300 * 300 * 3) to Float32Array = (3 * 300 * 300)
        const rearranged = rearrangeArray(resized, 300, 300, 3);

        // Run the model with the resized input buffer
        const outputs = model.runSync([rearranged]);
        
        // Get the prediction from the model outputs
        const prediction = getPrediction(outputs);

        // Convert the prediction to rock, paper, or scissors
        const convertedPrediction = convertPredictionToRPS(prediction);
        console.log(convertedPrediction);

        // Run the game logic
        gameLogic(convertedPrediction);
      }
    })

  }, [model, playing.value]);

  if (plugin.state !== "loaded" || model == null) {
    // Model is still loading.
    return (
      <View style={{
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
      }}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text style={styles.text}>Model is still loading</Text>
      </View>
    );
  }

  if (!hasPermission) {
    // Camera permissions are not granted yet.
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  const toggleCameraFacing = () => {
    if (facing.value === 'back') {
      if (frontDevice !== null) {
        facing.value = 'front';
        setDevice(frontDevice);
        return;
      } 
    } else if (facing.value === 'front') {
      if (backDevice !== null) {
        facing.value = 'back';
        setDevice(backDevice);
        return;
      }
      
    }
  }

  const handlePlayButtonPress = () => {
    if (!playing.value) {
      // Reset the game score
      gameScore.value = 0;
      // Reset the round
      round.value = 1;

      // Reset the round time
      roundTime.value = roundTimeLimit;

      // Reset the last computer move
      lastComputerMove.value = null;

      // Reset the last player move
      lastPlayerMove.value = null;

      // Reset the last winner
      lastWinner.value = null;

      // Reset the last predicted move
      lastPredictedMove.value = null;

    }
    // Add functionality for the Play button here
    playing.value = !playing.value;
  }

  if (device == null) {
    return null;
  }

  

  return (
    <View style={{ flex: 1, zIndex: 5}}>
      { nonSharedPlaying ? (
        <View style={{ flex:1, zIndex: 5}}>
          <View style={{ position: 'absolute', top: 10, left: 0, right: 0, height: '20%', justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0, 0, 0, 0.5)', zIndex: 5 }}>
            <Text style={{ fontSize: 20, fontWeight: 'bold', color: 'white' }}>Round {currentRound}</Text>
            <Text style={{ fontSize: 20, fontWeight: 'bold', color: 'white' }}>Score: {currentScore}</Text>
            <Text style={{ fontSize: 20, fontWeight: 'bold', color: 'white' }}>Time Remaining in Round: {currentRoundTime.toFixed(2)}s</Text>
            {currentLastComputerMove != null && (
              <Text style={{ fontSize: 20, fontWeight: 'bold', color: 'white' }}>Last Computer Move: {currentLastComputerMove}</Text>
            )}
            {currentLastPlayerMove != null && (
              <Text style={{ fontSize: 20, fontWeight: 'bold', color: 'white' }}>Last Player Move: {currentLastPlayerMove}</Text>
            )}
          </View>
          <View style={{ position: 'absolute', bottom: 100, width: "100%", height: 80, justifyContent: 'center', alignItems: 'center', zIndex: 5 }}>
            <Text style={{ fontSize: 24, fontWeight: 'bold', color: 'red' }}>Instant Prediction: {currentLastPredictedMove}</Text>
          </View>
        </View>
      ) : (
        <View style = {{ flex: 1, zIndex: 5}}>
          <View style={{ position: 'absolute', top: '50%', left: '50%', transform: [{ translateX: -200 }, { translateY: -200 }], width: 400, height: 400, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0, 0, 0, 0.5)', zIndex: 5 }}>
            <Text style={{ fontSize: 24, fontWeight: 'bold', color: 'white' }}>Rock, Paper, Scissors</Text>
            <Text style={{ fontSize: 24, fontWeight: 'bold', color: 'white' }}>Press Play to start</Text>
          </View>
          {currentLastWinner != null && (
          <View style={{ position: 'absolute', top: 10, left: 0, right: 0, height: '20%', justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0, 0, 0, 0.5)', zIndex: 5 }}>
              <Text style={{ fontSize: 24, fontWeight: 'bold', color: 'white' }}>Last Winner: {currentLastWinner}</Text>
              <Text style={{ fontSize: 24, fontWeight: 'bold', color: 'white' }}>Final Score: {currentScore}</Text>
          </View>
          )}
        </View>
        
      )}
      
      <View style={{ position: 'absolute', top: '50%', left: '50%', transform: [{ translateX: -200 }, { translateY: -200 }], width: 400, height: 400, borderColor: 'red', borderWidth: 2, zIndex: 10 }} />

      <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
            <Text style={styles.text}>Flip</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={handlePlayButtonPress}>
            
            <Text style={styles.text}> {nonSharedPlaying ? 'Stop' : 'Play'}</Text>
          </TouchableOpacity>
        </View>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
        
      />

    </View>
    
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  buttonContainer: {
    position: 'absolute',
    flexDirection: 'row',
    backgroundColor: 'transparent',
    bottom: 50,
    zIndex: 10,
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 10,
    borderRadius: 15,
    marginHorizontal: 5,
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
});
