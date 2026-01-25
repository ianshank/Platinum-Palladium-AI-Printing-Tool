import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Input } from './Input';

describe('Input', () => {
  describe('Rendering', () => {
    it('renders correctly with default props', () => {
      render(<Input placeholder="Enter text" />);
      expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
    });

    it('renders with custom className', () => {
      render(<Input className="custom-class" data-testid="input" />);
      expect(screen.getByTestId('input')).toHaveClass('custom-class');
    });

    it('renders with different types', () => {
      const { rerender } = render(<Input type="text" data-testid="input" />);
      expect(screen.getByTestId('input')).toHaveAttribute('type', 'text');

      rerender(<Input type="password" data-testid="input" />);
      expect(screen.getByTestId('input')).toHaveAttribute('type', 'password');

      rerender(<Input type="email" data-testid="input" />);
      expect(screen.getByTestId('input')).toHaveAttribute('type', 'email');
    });
  });

  describe('Error State', () => {
    it('displays error message when error prop is provided', () => {
      render(<Input error="This field is required" />);
      expect(screen.getByRole('alert')).toHaveTextContent('This field is required');
    });

    it('applies error styling when error prop is provided', () => {
      render(<Input error="Error" data-testid="input" />);
      expect(screen.getByTestId('input')).toHaveClass('border-destructive');
    });

    it('sets aria-invalid to true when error is present', () => {
      render(<Input error="Error" data-testid="input" />);
      expect(screen.getByTestId('input')).toHaveAttribute('aria-invalid', 'true');
    });

    it('links error message with aria-describedby', () => {
      render(<Input id="test-input" error="Error message" />);
      const input = screen.getByRole('textbox');
      const errorId = input.getAttribute('aria-describedby');
      expect(errorId).toBe('test-input-error');
      expect(screen.getByRole('alert')).toHaveAttribute('id', 'test-input-error');
    });

    it('generates unique ID when not provided', () => {
      render(<Input error="Error message" />);
      const input = screen.getByRole('textbox');
      expect(input.getAttribute('aria-describedby')).toBeDefined();
      expect(input.getAttribute('id')).toBeDefined();
    });
  });

  describe('Left and Right Elements', () => {
    it('renders left element', () => {
      render(
        <Input
          leftElement={<span data-testid="left-icon">L</span>}
          data-testid="input"
        />
      );
      expect(screen.getByTestId('left-icon')).toBeInTheDocument();
    });

    it('renders right element', () => {
      render(
        <Input
          rightElement={<span data-testid="right-icon">R</span>}
          data-testid="input"
        />
      );
      expect(screen.getByTestId('right-icon')).toBeInTheDocument();
    });

    it('adds padding when left element is present', () => {
      render(
        <Input
          leftElement={<span>L</span>}
          data-testid="input"
        />
      );
      expect(screen.getByTestId('input')).toHaveClass('pl-10');
    });

    it('adds padding when right element is present', () => {
      render(
        <Input
          rightElement={<span>R</span>}
          data-testid="input"
        />
      );
      expect(screen.getByTestId('input')).toHaveClass('pr-10');
    });
  });

  describe('Disabled State', () => {
    it('can be disabled', () => {
      render(<Input disabled data-testid="input" />);
      expect(screen.getByTestId('input')).toBeDisabled();
    });
  });

  describe('User Interaction', () => {
    it('accepts user input', async () => {
      const user = userEvent.setup();
      render(<Input data-testid="input" />);

      const input = screen.getByTestId('input');
      await user.type(input, 'Hello');

      expect(input).toHaveValue('Hello');
    });

    it('can be focused', async () => {
      const user = userEvent.setup();
      render(<Input data-testid="input" />);

      const input = screen.getByTestId('input');
      await user.click(input);

      expect(input).toHaveFocus();
    });
  });

  describe('Accessibility', () => {
    it('has no aria-invalid when no error', () => {
      render(<Input data-testid="input" />);
      expect(screen.getByTestId('input')).toHaveAttribute('aria-invalid', 'false');
    });

    it('has no aria-describedby when no error', () => {
      render(<Input data-testid="input" />);
      expect(screen.getByTestId('input')).not.toHaveAttribute('aria-describedby');
    });
  });

  describe('Ref Forwarding', () => {
    it('forwards ref to input element', () => {
      const ref = { current: null as HTMLInputElement | null };
      render(<Input ref={ref} data-testid="input" />);
      expect(ref.current).toBeInstanceOf(HTMLInputElement);
    });
  });
});
