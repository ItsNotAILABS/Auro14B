# Auro Sovereign Mobile

Expo SDK 54 client for physical-device testing in Expo Go. It connects to the
receipt-rich Auro API, identifies itself as the Expo surface, and sends bounded
accelerometer state as an optional native sense. It never enables execution.

```bash
npm install
npx expo start
```

Install Expo Go from <https://expo.dev/go>, scan the terminal QR code, and set
the API field to this computer's LAN address (for example,
`http://192.168.1.10:8090`). A phone cannot reach the computer through
`127.0.0.1`.

Expo Go is the fast development surface. Production distribution should use an
Expo development build/EAS build so native modules, signing, update ownership,
and store testing are controlled explicitly.

Security note: the pinned SDK-54 tree currently reports 11 moderate transitive
npm advisories. Expo's suggested automated remediation upgrades to SDK 57,
which is outside this Expo Go compatibility target. `expo-doctor` passes 18/18;
re-audit and move to a development build/current SDK before production release.
