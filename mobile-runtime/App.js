import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { Accelerometer } from 'expo-sensors';
import { ActivityIndicator, Linking, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

const EXPO_GO_URL = 'https://expo.dev/go';

export default function App() {
  const [endpoint, setEndpoint] = useState('http://192.168.1.10:8090');
  const [objective, setObjective] = useState('Report the Auro, MESIE, and Sovereign runtime state.');
  const [answer, setAnswer] = useState('Connect to the Auro API on your LAN. Do not use localhost from a physical phone.');
  const [motion, setMotion] = useState({ x: 0, y: 0, z: 0 });
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    Accelerometer.setUpdateInterval(500);
    const subscription = Accelerometer.addListener(setMotion);
    return () => subscription.remove();
  }, []);

  async function askAuro() {
    setBusy(true);
    try {
      const response = await fetch(`${endpoint.replace(/\/$/, '')}/v1/respond`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          message: objective,
          execute: false,
          senses: { accelerometer: motion },
          client: { surface: 'expo-go', sdk: 54 },
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.error?.message || `HTTP ${response.status}`);
      setAnswer(JSON.stringify({ answer: payload.answer, confidence: payload.confidence, receipt: payload.receipt }, null, 2));
    } catch (error) {
      setAnswer(`Connection failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.page}>
        <Text style={styles.eyebrow}>SOVEREIGN MULTI-DEVICE RUNTIME</Text>
        <Text style={styles.title}>Auro mobile organism</Text>
        <Text style={styles.copy}>Expo Go client for Auro inference, MESIE compute, agents, receipts, and native device senses.</Text>

        <View style={styles.panel}>
          <Text style={styles.label}>Auro API endpoint</Text>
          <TextInput value={endpoint} onChangeText={setEndpoint} autoCapitalize="none" autoCorrect={false} style={styles.input} />
          <Text style={styles.hint}>Use this computer's LAN address and port 8090.</Text>
          <Text style={styles.label}>Objective</Text>
          <TextInput value={objective} onChangeText={setObjective} multiline style={[styles.input, styles.objective]} />
          <Pressable onPress={askAuro} disabled={busy} style={styles.primary}>
            {busy ? <ActivityIndicator color="#17130b" /> : <Text style={styles.primaryText}>Ask Auro</Text>}
          </Pressable>
        </View>

        <View style={styles.panel}>
          <Text style={styles.label}>Accelerometer sense</Text>
          <Text style={styles.mono}>x {motion.x.toFixed(3)}  y {motion.y.toFixed(3)}  z {motion.z.toFixed(3)}</Text>
          <Text style={styles.label}>Receipt-backed response</Text>
          <Text selectable style={styles.output}>{answer}</Text>
        </View>

        <Pressable onPress={() => Linking.openURL(EXPO_GO_URL)} style={styles.secondary}>
          <Text style={styles.secondaryText}>Download Expo Go</Text>
        </Pressable>
        <Text style={styles.hint}>Expo Go is for rapid testing. Use an Expo development build for production delivery.</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#10110f' }, page: { padding: 22, gap: 16 },
  eyebrow: { color: '#d9a441', fontSize: 11, fontWeight: '800', letterSpacing: 2, marginTop: 18 },
  title: { color: '#f1eee4', fontSize: 44, lineHeight: 46, fontWeight: '300' },
  copy: { color: '#aaa99e', fontSize: 16, lineHeight: 24 },
  panel: { backgroundColor: '#191b17', borderColor: '#34382f', borderWidth: 1, padding: 18, gap: 10 },
  label: { color: '#d9a441', fontSize: 11, fontWeight: '700', letterSpacing: 1, textTransform: 'uppercase', marginTop: 5 },
  input: { color: '#f1eee4', backgroundColor: '#0b0c0a', borderColor: '#3d4137', borderWidth: 1, padding: 12, fontSize: 15 },
  objective: { minHeight: 110, textAlignVertical: 'top' }, hint: { color: '#777b70', fontSize: 12, lineHeight: 17 },
  primary: { alignItems: 'center', backgroundColor: '#d9a441', padding: 14, marginTop: 6 }, primaryText: { color: '#17130b', fontWeight: '800' },
  secondary: { alignItems: 'center', borderColor: '#d9a441', borderWidth: 1, padding: 14 }, secondaryText: { color: '#f1eee4', fontWeight: '700' },
  mono: { color: '#c9cbbf', fontFamily: 'monospace' }, output: { color: '#d7d7cf', backgroundColor: '#0b0c0a', padding: 12, fontFamily: 'monospace', lineHeight: 19 },
});
