// app/index.tsx

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Link } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';

export default function Home() {
  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">Rock Paper Scissors</ThemedText>
      <ThemedView style={styles.viewBelow}>
        <ThemedText>Welcome to the Rock Paper Scissors Mobile Game</ThemedText>
        <Link href="/cameraScreen"><ThemedText type="link">Play the Game</ThemedText></Link>
      </ThemedView>
      <ThemedView style={styles.footer}>
        <ThemedText>Created by: </ThemedText>
        <Link href="https://github.com/b-rbmp/"><ThemedText type="link">Bernardo Ribeiro</ThemedText></Link>
        <Link href="https://github.com/RobCTs/"><ThemedText type="link">Roberta Chissich</ThemedText></Link>
      </ThemedView>

    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    gap: 80,
    paddingBottom: 100,
    alignItems: 'center',
  },
  viewBelow: {
    gap: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  footer: {
    position: 'absolute',
    bottom: 20,
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 5,
  },
});

