import { expect, test } from '@rstest/core';
import { render, screen } from '@testing-library/react';
import '../src/i18n';
import App from '../src/App';

test('renders the main page', async () => {
  render(<App />);
  expect(
    await screen.findByText(/Neural Network from Scratch|Mạng Nơ-ron Từ Đầu/),
  ).toBeInTheDocument();
});
