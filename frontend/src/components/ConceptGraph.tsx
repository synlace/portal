import { useMemo, useCallback, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  type Node,
  type Edge,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from '@dagrejs/dagre';

// ─── Types ──────────────────────────────────────────────────────────────────
export type ConceptNodeType = 'service' | 'infrastructure' | 'cross-cutting';

export interface ConceptNode {
  id: string;
  label: string;
  parent: string | null;
  type?: ConceptNodeType;
  appliesTo?: string[];
  connectedTo?: string[];
}

interface ConceptRelationship {
  from: string;
  to: string;
  label: string;
}

interface ConceptGraphData {
  concepts: ConceptNode[];
  specs: Record<string, string[]>;
  relationships: ConceptRelationship[];
}

interface ConceptGraphProps {
  graph: ConceptGraphData | null;
  activeNode: string | null;
  onNodeClick: (nodeId: string) => void;
  isFullscreen?: boolean;
}

// ─── Dagre layout helper ────────────────────────────────────────────────────
function getLayoutedElements(concepts: ConceptNode[], relationships: ConceptRelationship[]) {
  const mainNodes = concepts.filter(c => c.type !== 'cross-cutting');
  const crossCutting = concepts.filter(c => c.type === 'cross-cutting');

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', nodesep: 50, ranksep: 70, marginx: 30, marginy: 30 });

  // Add main nodes
  mainNodes.forEach(c => {
    const w = c.type === 'infrastructure' ? 110 : 130;
    const h = c.type === 'infrastructure' ? 38 : 44;
    g.setNode(c.id, { width: w, height: h });
  });

  // Add cross-cutting nodes (smaller, positioned later)
  crossCutting.forEach(c => {
    g.setNode(c.id, { width: 90, height: 32 });
  });

  // Layout edges: relationships + infrastructure connections
  relationships.forEach(r => g.setEdge(r.from, r.to));
  concepts.filter(c => c.type === 'infrastructure' && c.connectedTo).forEach(c => {
    c.connectedTo!.forEach(svc => g.setEdge(svc, c.id));
  });

  // Layout cross-cutting edges
  crossCutting.forEach(cc => {
    (cc.appliesTo || []).forEach(target => {
      if (g.node(target)) g.setEdge(cc.id, target);
    });
  });

  dagre.layout(g);

  // Find the rightmost x of main nodes to position cross-cutting to the side
  const mainRightX = mainNodes.length > 0
    ? Math.max(...mainNodes.map(c => (g.node(c.id)?.x || 0) + 65))
    : 400;

  const nodes: Node[] = concepts.map(c => {
    const pos = g.node(c.id);
    if (c.type === 'cross-cutting') {
      const idx = crossCutting.indexOf(c);
      return {
        id: c.id,
        position: { x: mainRightX + 40, y: 40 + idx * 55 },
        data: { label: c.label, nodeType: c.type },
        type: 'crossCuttingNode',
      };
    }
    return {
      id: c.id,
      position: { x: pos.x - (c.type === 'infrastructure' ? 55 : 65), y: pos.y - (c.type === 'infrastructure' ? 19 : 22) },
      data: { label: c.label, nodeType: c.type },
      type: c.type === 'infrastructure' ? 'infraNode' : 'serviceNode',
    };
  });

  // Build edges
  const edgeSet = new Set<string>();
  const edges: Edge[] = [];

  const addEdge = (from: string, to: string, label: string, style: Record<string, unknown> = {}) => {
    const key = `${from}->${to}`;
    if (edgeSet.has(key)) return;
    edgeSet.add(key);
    edges.push({ id: key, source: from, target: to, label, style, type: 'smoothstep' });
  };

  relationships.forEach(r => addEdge(r.from, r.to, r.label));

  // Infrastructure connections (dashed amber)
  concepts.filter(c => c.type === 'infrastructure' && c.connectedTo).forEach(c => {
    c.connectedTo!.forEach(svc => addEdge(svc, c.id, '', { stroke: '#92400e', strokeWidth: 1, strokeDasharray: '4,4' }));
  });

  // Cross-cutting connections (dashed violet)
  crossCutting.forEach(cc => {
    (cc.appliesTo || []).forEach(target => addEdge(cc.id, target, '', { stroke: '#7c3aed', strokeWidth: 1, strokeDasharray: '3,3' }));
  });

  return { nodes, edges };
}

// ─── Custom Node Components ─────────────────────────────────────────────────
interface NodeData {
  label: string;
  nodeType: ConceptNodeType;
  specCount?: number;
  isActive?: boolean;
  [key: string]: unknown;
}

function ServiceNode({ data }: { data: NodeData }) {
  return (
    <div className={`
      px-3 py-2 rounded-lg text-center font-mono text-xs font-bold transition-all cursor-pointer select-none
      ${data.isActive
        ? 'bg-gray-800 border border-violet-500 text-violet-300 shadow-md shadow-violet-500/10'
        : 'bg-gray-900 border border-gray-700 text-gray-300 hover:border-gray-500 hover:bg-gray-800'
      }
    `}>
      {data.label}
      {data.specCount != null && data.specCount > 0 && (
        <div className="text-[7px] text-gray-500 mt-0.5 font-normal">
          {data.specCount} spec{data.specCount > 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}

function InfraNode({ data }: { data: NodeData }) {
  return (
    <div className={`
      px-3 py-1.5 rounded-lg text-center font-mono text-[10px] font-bold transition-all cursor-pointer select-none
      ${data.isActive
        ? 'bg-amber-950/70 border border-amber-600 text-amber-200 shadow-md shadow-amber-500/10'
        : 'bg-amber-950/50 border border-amber-800/60 text-amber-300 hover:border-amber-600'
      }
    `}>
      {data.label}
      {data.specCount != null && data.specCount > 0 && (
        <div className="text-[7px] text-amber-500/60 mt-0.5 font-normal">
          {data.specCount} spec{data.specCount > 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}

function CrossCuttingNode({ data }: { data: NodeData }) {
  return (
    <div className={`
      px-3 py-1.5 rounded text-center font-mono text-[10px] font-bold transition-all cursor-pointer select-none
      ${data.isActive
        ? 'bg-violet-950/70 border border-violet-500 border-dashed text-violet-200 shadow-md shadow-violet-500/10'
        : 'bg-violet-950/50 border border-violet-700/60 border-dashed text-violet-300 hover:border-violet-500'
      }
    `}>
      ◇ {data.label}
      {data.specCount != null && data.specCount > 0 && (
        <div className="text-[7px] text-violet-400/60 mt-0.5 font-normal">
          {data.specCount} spec{data.specCount > 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}

// ─── Node Types Registration ────────────────────────────────────────────────
const nodeTypes: NodeTypes = {
  serviceNode: ServiceNode,
  infraNode: InfraNode,
  crossCuttingNode: CrossCuttingNode,
};

// ─── MiniMap node color helper ──────────────────────────────────────────────
function minimapNodeColor(node: Node) {
  if (node.type === 'infraNode') return '#92400e';
  if (node.type === 'crossCuttingNode') return '#6d28d9';
  return '#374151';
}

// ─── Inner Component (needs ReactFlowProvider context) ──────────────────────
function ConceptGraphInner({ graph, activeNode, onNodeClick, isFullscreen }: ConceptGraphProps) {
  const { fitView } = useReactFlow();

  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(() => {
    if (!graph || graph.concepts.length === 0) return { nodes: [], edges: [] };
    return getLayoutedElements(graph.concepts, graph.relationships);
  }, [graph]);

  const enrichedNodes = useMemo(() => {
    return layoutedNodes.map(n => ({
      ...n,
      data: {
        ...n.data,
        specCount: graph?.specs[n.id]?.length || 0,
        isActive: activeNode === n.id,
      },
    }));
  }, [layoutedNodes, graph, activeNode]);

  const [nodes, setNodes, onNodesChange] = useNodesState(enrichedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges);

  useEffect(() => { setNodes(enrichedNodes); }, [enrichedNodes, setNodes]);
  useEffect(() => { setEdges(layoutedEdges); }, [layoutedEdges, setEdges]);

  useEffect(() => {
    if (isFullscreen) {
      const timer = setTimeout(() => fitView({ padding: 0.3, duration: 300 }), 50);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [isFullscreen, fitView]);

  const handleNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    onNodeClick(node.id);
  }, [onNodeClick]);

  if (!graph || graph.concepts.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-6 text-gray-600">
        <div className="text-2xl mb-2 opacity-30">📊</div>
        <p className="text-[11px]">No concept graph yet.</p>
        <p className="text-[9px] text-gray-700 mt-1">Run onboarding to generate one.</p>
      </div>
    );
  }

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={true}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={true}
        panOnScroll={true}
        zoomOnScroll={true}
        zoomOnPinch={true}
        zoomOnDoubleClick={true}
        minZoom={0.2}
        maxZoom={4}
      >
        <Background color="#1f2937" gap={20} size={1} />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor={minimapNodeColor}
          maskColor="rgba(3, 7, 18, 0.8)"
          style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 8, width: 120, height: 80 }}
          pannable={false}
          zoomable={false}
        />
      </ReactFlow>
    </div>
  );
}

// ─── Exported Component (wrapped in ReactFlowProvider) ──────────────────────
export default function ConceptGraph(props: ConceptGraphProps) {
  return (
    <ReactFlowProvider>
      <ConceptGraphInner {...props} />
    </ReactFlowProvider>
  );
}
