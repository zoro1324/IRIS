import 'package:flutter_test/flutter_test.dart';
import 'package:iris/main.dart';

void main() {
  testWidgets('App loads smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const IrisApp());
    expect(find.text('IRIS'), findsOneWidget);
  });
}
