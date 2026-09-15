import { DataProvider } from './context/DataContext';
import { Shell } from './components/layout/Shell';

export function App() {
  return (
    <DataProvider>
      <Shell />
    </DataProvider>
  );
}

export default App;
