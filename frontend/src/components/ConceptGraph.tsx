import { useMemo, useCallback, useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from '@dagrejs/dagre';

// ─── Types ──────────────────────────────────────────────────────────────────
interface ConceptNode {
  id: string;
  label: string;
  parent: string | null;
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
}

// ─── Dagre layout helper ────────────────────────────────────────────────────
function getLayoutedElements(concepts: ConceptNode[], relationships: ConceptRelationship[]) {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', nodesep: 60, ranksep: 80, marginx: 20, marginy: 20 });

  const parentEdges = concepts
    .filter(c => c.parent)
    .map(c => ({ source: c.parent!, target: c.id }));

  concepts.forEach(c => {
    g.setNode(c.id, { width: 140, height: 50 });
  });

  parentEdges.forEach(e => {
    g.setEdge(e.source, e.target);
  });

  dagre.layout(g);

  const nodes: Node[] = concepts.map(c => {
    const pos = g.node(c.id);
    return {
      id: c.id,
      position: { x: pos.x - 70, y: pos.y - 25 },
      data: { label: c.label, isRoot: !c.parent },
      type: 'conceptNode',
    };
  });

  const edgeSet = new Set<string>();
  const edges: Edge[] = [];

  const addEdge = (from: string, to: string, label: string, dashed: boolean) => {
    const key = `${from}->${to}`;
    if (edgeSet.has(key)) return;
    edgeSet.add(key);
    edges.push({
      id: key,
      source: from,
      target: to,
      label,
      animated: false,
      style: { stroke: dashed ? '#374151' : '#4b5563', strokeWidth: dashed ? 1 : 1.5, strokeDasharray: dashed ? '5,5' : undefined },
      type: 'smoothstep',
    });
  };

  if (relationships.length > 0) {
    relationships.forEach(r => addEdge(r.from, r.to, r.label, false));
  }
  parentEdges.forEach(e => addEdge(e.source, e.target, '', true));

  return { nodes, edges };
}

// ─── Custom Node Component ──────────────────────────────────────────────────
interface ConceptNodeData {
  label: string;
  isRoot: boolean;
  specCount?: number;
  isActive?: boolean;
  [key: string]: unknown;
}

function ConceptNodeComponent({ data }: { data: ConceptNodeData; selected?: boolean }) {
  const isRoot = data.isRoot;
  const isActive = data.isActive;
  return (
    <div
      className={`
        px-4 py-2 rounded-lg text-center font-mono text-xs font-bold transition-all cursor-pointer
        ${isRoot
          ? 'bg-violet-950/80 border-2 border-violet-500 text-violet-200 shadow-lg shadow-violet-500/10'
          : isActive
            ? 'bg-gray-800 border border-violet-500 text-violet-300 shadow-md shadow-violet-500/10'
            : 'bg-gray-900 border border-gray-700 text-gray-300 hover:border-gray-500 hover:bg-gray-800'
        }
      `}
    >
      {data.label}
      {data.specCount != null && data.specCount > 0 && (
        <div className="text-[8px] text-gray-500 mt-0.5 font-normal">
          {data.specCount} spec{data.specCount > 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}

// ─── Node Types Registration ────────────────────────────────────────────────
const nodeTypes: NodeTypes = {
  conceptNode: ConceptNodeComponent,
};

// ─── Component ──────────────────────────────────────────────────────────────
export default function ConceptGraph({ graph, activeNode, onNodeClick }: ConceptGraphProps) {
  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(() => {
    if (!graph || graph.concepts.length === 0) return { nodes: [], edges: [] };
    return getLayoutedElements(graph.concepts, graph.relationships);
  }, [graph]);

  // Inject spec counts and active state into node data
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
  const [edges, , onEdgesChange] = useEdgesState(layoutedEdges);

  // Sync nodes when enrichedNodes change
  useEffect(() => {
    setNodes(enrichedNodes);
  }, [enrichedNodes, setNodes]);

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
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        minZoom={0.5}
        maxZoom={2}
      >
        <Background color="#1f2937" gap={20} size={1} />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor="#374151"
          maskColor="rgba(3, 7, 18, 0.8)"
          style={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 8 }}
        />
      </ReactFlow>
    </div>
  );
}
